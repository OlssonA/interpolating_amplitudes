module     p2_gg_httbar_d142h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d142h4l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd142h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(56) :: acd142
      complex(ki) :: brack
      acd142(1)=dotproduct(qshift,qshift)
      acd142(2)=abb142(18)
      acd142(3)=dotproduct(qshift,spvak1e2)
      acd142(4)=dotproduct(qshift,spvae2k1)
      acd142(5)=abb142(90)
      acd142(6)=dotproduct(qshift,spvae2k2)
      acd142(7)=abb142(14)
      acd142(8)=dotproduct(qshift,spvae2l4)
      acd142(9)=abb142(24)
      acd142(10)=abb142(28)
      acd142(11)=dotproduct(qshift,spvak2e2)
      acd142(12)=abb142(22)
      acd142(13)=dotproduct(qshift,spval5e2)
      acd142(14)=abb142(19)
      acd142(15)=abb142(13)
      acd142(16)=abb142(12)
      acd142(17)=abb142(41)
      acd142(18)=dotproduct(qshift,spval4e2)
      acd142(19)=abb142(43)
      acd142(20)=dotproduct(qshift,spvae1e2)
      acd142(21)=abb142(38)
      acd142(22)=abb142(17)
      acd142(23)=abb142(25)
      acd142(24)=abb142(94)
      acd142(25)=abb142(89)
      acd142(26)=abb142(72)
      acd142(27)=dotproduct(qshift,spvae2l5)
      acd142(28)=abb142(23)
      acd142(29)=dotproduct(qshift,spvae2e1)
      acd142(30)=abb142(21)
      acd142(31)=abb142(20)
      acd142(32)=abb142(86)
      acd142(33)=abb142(71)
      acd142(34)=abb142(104)
      acd142(35)=abb142(16)
      acd142(36)=abb142(53)
      acd142(37)=abb142(78)
      acd142(38)=abb142(70)
      acd142(39)=dotproduct(qshift,spval3e2)
      acd142(40)=abb142(15)
      acd142(41)=dotproduct(qshift,spvae2l3)
      acd142(42)=abb142(27)
      acd142(43)=abb142(59)
      acd142(44)=acd142(12)*acd142(4)
      acd142(45)=acd142(16)*acd142(6)
      acd142(46)=acd142(23)*acd142(8)
      acd142(47)=acd142(28)*acd142(27)
      acd142(48)=acd142(30)*acd142(29)
      acd142(44)=-acd142(31)+acd142(48)+acd142(47)+acd142(46)+acd142(45)+acd142&
      &(44)
      acd142(44)=acd142(11)*acd142(44)
      acd142(45)=-acd142(27)*acd142(5)
      acd142(46)=acd142(14)*acd142(4)
      acd142(47)=acd142(17)*acd142(6)
      acd142(48)=acd142(24)*acd142(8)
      acd142(49)=acd142(32)*acd142(29)
      acd142(45)=-acd142(33)+acd142(49)+acd142(48)+acd142(47)+acd142(46)+acd142&
      &(45)
      acd142(45)=acd142(13)*acd142(45)
      acd142(46)=-acd142(5)*acd142(4)
      acd142(47)=acd142(7)*acd142(6)
      acd142(48)=acd142(9)*acd142(8)
      acd142(46)=-acd142(10)+acd142(48)+acd142(47)+acd142(46)
      acd142(46)=acd142(3)*acd142(46)
      acd142(47)=acd142(21)*acd142(6)
      acd142(48)=acd142(25)*acd142(8)
      acd142(49)=acd142(35)*acd142(29)
      acd142(47)=-acd142(38)+acd142(49)+acd142(48)+acd142(47)
      acd142(47)=acd142(20)*acd142(47)
      acd142(48)=-acd142(5)*acd142(8)
      acd142(49)=acd142(19)*acd142(6)
      acd142(48)=-acd142(37)+acd142(49)+acd142(48)
      acd142(48)=acd142(18)*acd142(48)
      acd142(49)=acd142(2)*acd142(1)
      acd142(50)=-acd142(15)*acd142(4)
      acd142(51)=-acd142(22)*acd142(6)
      acd142(52)=-acd142(26)*acd142(8)
      acd142(53)=-acd142(34)*acd142(27)
      acd142(54)=-acd142(36)*acd142(29)
      acd142(55)=-acd142(40)*acd142(39)
      acd142(56)=-acd142(42)*acd142(41)
      brack=acd142(43)+acd142(44)+acd142(45)+acd142(46)+acd142(47)+acd142(48)+a&
      &cd142(49)+acd142(50)+acd142(51)+acd142(52)+acd142(53)+acd142(54)+acd142(&
      &55)+acd142(56)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd142h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(70) :: acd142
      complex(ki) :: brack
      acd142(1)=qshift(iv1)
      acd142(2)=abb142(18)
      acd142(3)=spvak1e2(iv1)
      acd142(4)=dotproduct(qshift,spvae2k1)
      acd142(5)=abb142(90)
      acd142(6)=dotproduct(qshift,spvae2k2)
      acd142(7)=abb142(14)
      acd142(8)=dotproduct(qshift,spvae2l4)
      acd142(9)=abb142(24)
      acd142(10)=abb142(28)
      acd142(11)=spvae2k1(iv1)
      acd142(12)=dotproduct(qshift,spvak1e2)
      acd142(13)=dotproduct(qshift,spvak2e2)
      acd142(14)=abb142(22)
      acd142(15)=dotproduct(qshift,spval5e2)
      acd142(16)=abb142(19)
      acd142(17)=abb142(13)
      acd142(18)=spvae2k2(iv1)
      acd142(19)=abb142(12)
      acd142(20)=abb142(41)
      acd142(21)=dotproduct(qshift,spval4e2)
      acd142(22)=abb142(43)
      acd142(23)=dotproduct(qshift,spvae1e2)
      acd142(24)=abb142(38)
      acd142(25)=abb142(17)
      acd142(26)=spvae2l4(iv1)
      acd142(27)=abb142(25)
      acd142(28)=abb142(94)
      acd142(29)=abb142(89)
      acd142(30)=abb142(72)
      acd142(31)=spvak2e2(iv1)
      acd142(32)=dotproduct(qshift,spvae2l5)
      acd142(33)=abb142(23)
      acd142(34)=dotproduct(qshift,spvae2e1)
      acd142(35)=abb142(21)
      acd142(36)=abb142(20)
      acd142(37)=spval5e2(iv1)
      acd142(38)=abb142(86)
      acd142(39)=abb142(71)
      acd142(40)=spvae2l5(iv1)
      acd142(41)=abb142(104)
      acd142(42)=spvae2e1(iv1)
      acd142(43)=abb142(16)
      acd142(44)=abb142(53)
      acd142(45)=spval4e2(iv1)
      acd142(46)=abb142(78)
      acd142(47)=spvae1e2(iv1)
      acd142(48)=abb142(70)
      acd142(49)=spval3e2(iv1)
      acd142(50)=abb142(15)
      acd142(51)=spvae2l3(iv1)
      acd142(52)=abb142(27)
      acd142(53)=acd142(11)*acd142(12)
      acd142(54)=acd142(3)*acd142(4)
      acd142(55)=acd142(15)*acd142(40)
      acd142(56)=acd142(8)*acd142(45)
      acd142(57)=acd142(37)*acd142(32)
      acd142(58)=acd142(26)*acd142(21)
      acd142(53)=acd142(58)+acd142(57)+acd142(56)+acd142(55)+acd142(53)+acd142(&
      &54)
      acd142(53)=acd142(5)*acd142(53)
      acd142(54)=-acd142(32)*acd142(33)
      acd142(55)=-acd142(34)*acd142(35)
      acd142(56)=-acd142(4)*acd142(14)
      acd142(57)=-acd142(8)*acd142(27)
      acd142(58)=-acd142(6)*acd142(19)
      acd142(54)=acd142(58)+acd142(57)+acd142(56)+acd142(55)+acd142(36)+acd142(&
      &54)
      acd142(54)=acd142(31)*acd142(54)
      acd142(55)=-acd142(21)*acd142(22)
      acd142(56)=-acd142(23)*acd142(24)
      acd142(57)=-acd142(12)*acd142(7)
      acd142(58)=-acd142(15)*acd142(20)
      acd142(59)=-acd142(13)*acd142(19)
      acd142(55)=acd142(59)+acd142(58)+acd142(57)+acd142(56)+acd142(25)+acd142(&
      &55)
      acd142(55)=acd142(18)*acd142(55)
      acd142(56)=-acd142(34)*acd142(38)
      acd142(57)=-acd142(4)*acd142(16)
      acd142(58)=-acd142(8)*acd142(28)
      acd142(59)=-acd142(6)*acd142(20)
      acd142(56)=acd142(59)+acd142(58)+acd142(57)+acd142(39)+acd142(56)
      acd142(56)=acd142(37)*acd142(56)
      acd142(57)=-acd142(23)*acd142(29)
      acd142(58)=-acd142(12)*acd142(9)
      acd142(59)=-acd142(15)*acd142(28)
      acd142(60)=-acd142(13)*acd142(27)
      acd142(57)=acd142(60)+acd142(59)+acd142(58)+acd142(30)+acd142(57)
      acd142(57)=acd142(26)*acd142(57)
      acd142(58)=-acd142(40)*acd142(33)
      acd142(59)=-acd142(42)*acd142(35)
      acd142(60)=-acd142(11)*acd142(14)
      acd142(58)=acd142(60)+acd142(58)+acd142(59)
      acd142(58)=acd142(13)*acd142(58)
      acd142(59)=-acd142(45)*acd142(22)
      acd142(60)=-acd142(47)*acd142(24)
      acd142(61)=-acd142(3)*acd142(7)
      acd142(59)=acd142(61)+acd142(59)+acd142(60)
      acd142(59)=acd142(6)*acd142(59)
      acd142(60)=-acd142(42)*acd142(38)
      acd142(61)=-acd142(11)*acd142(16)
      acd142(60)=acd142(60)+acd142(61)
      acd142(60)=acd142(15)*acd142(60)
      acd142(61)=-acd142(47)*acd142(29)
      acd142(62)=-acd142(3)*acd142(9)
      acd142(61)=acd142(61)+acd142(62)
      acd142(61)=acd142(8)*acd142(61)
      acd142(62)=acd142(51)*acd142(52)
      acd142(63)=acd142(49)*acd142(50)
      acd142(64)=acd142(1)*acd142(2)
      acd142(65)=acd142(45)*acd142(46)
      acd142(66)=acd142(40)*acd142(41)
      acd142(67)=-acd142(34)*acd142(43)
      acd142(67)=acd142(48)+acd142(67)
      acd142(67)=acd142(47)*acd142(67)
      acd142(68)=-acd142(23)*acd142(43)
      acd142(68)=acd142(44)+acd142(68)
      acd142(68)=acd142(42)*acd142(68)
      acd142(69)=acd142(11)*acd142(17)
      acd142(70)=acd142(3)*acd142(10)
      brack=acd142(53)+acd142(54)+acd142(55)+acd142(56)+acd142(57)+acd142(58)+a&
      &cd142(59)+acd142(60)+acd142(61)+acd142(62)+acd142(63)-2.0_ki*acd142(64)+&
      &acd142(65)+acd142(66)+acd142(67)+acd142(68)+acd142(69)+acd142(70)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd142h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(49) :: acd142
      complex(ki) :: brack
      acd142(1)=d(iv1,iv2)
      acd142(2)=abb142(18)
      acd142(3)=spvak1e2(iv1)
      acd142(4)=spvae2k1(iv2)
      acd142(5)=abb142(90)
      acd142(6)=spvae2k2(iv2)
      acd142(7)=abb142(14)
      acd142(8)=spvae2l4(iv2)
      acd142(9)=abb142(24)
      acd142(10)=spvak1e2(iv2)
      acd142(11)=spvae2k1(iv1)
      acd142(12)=spvae2k2(iv1)
      acd142(13)=spvae2l4(iv1)
      acd142(14)=spvak2e2(iv2)
      acd142(15)=abb142(22)
      acd142(16)=spval5e2(iv2)
      acd142(17)=abb142(19)
      acd142(18)=spvak2e2(iv1)
      acd142(19)=spval5e2(iv1)
      acd142(20)=abb142(12)
      acd142(21)=abb142(41)
      acd142(22)=spval4e2(iv2)
      acd142(23)=abb142(43)
      acd142(24)=spvae1e2(iv2)
      acd142(25)=abb142(38)
      acd142(26)=spval4e2(iv1)
      acd142(27)=spvae1e2(iv1)
      acd142(28)=abb142(25)
      acd142(29)=abb142(94)
      acd142(30)=abb142(89)
      acd142(31)=spvae2l5(iv2)
      acd142(32)=abb142(23)
      acd142(33)=spvae2e1(iv2)
      acd142(34)=abb142(21)
      acd142(35)=spvae2l5(iv1)
      acd142(36)=spvae2e1(iv1)
      acd142(37)=abb142(86)
      acd142(38)=abb142(16)
      acd142(39)=-acd142(10)*acd142(11)
      acd142(40)=-acd142(3)*acd142(4)
      acd142(41)=-acd142(19)*acd142(31)
      acd142(42)=-acd142(16)*acd142(35)
      acd142(43)=-acd142(13)*acd142(22)
      acd142(44)=-acd142(8)*acd142(26)
      acd142(39)=acd142(44)+acd142(43)+acd142(42)+acd142(41)+acd142(39)+acd142(&
      &40)
      acd142(39)=acd142(5)*acd142(39)
      acd142(40)=acd142(22)*acd142(23)
      acd142(41)=acd142(24)*acd142(25)
      acd142(42)=acd142(10)*acd142(7)
      acd142(43)=acd142(16)*acd142(21)
      acd142(44)=acd142(14)*acd142(20)
      acd142(40)=acd142(44)+acd142(43)+acd142(42)+acd142(40)+acd142(41)
      acd142(40)=acd142(12)*acd142(40)
      acd142(41)=acd142(23)*acd142(26)
      acd142(42)=acd142(27)*acd142(25)
      acd142(43)=acd142(3)*acd142(7)
      acd142(44)=acd142(19)*acd142(21)
      acd142(45)=acd142(18)*acd142(20)
      acd142(41)=acd142(45)+acd142(44)+acd142(43)+acd142(41)+acd142(42)
      acd142(41)=acd142(6)*acd142(41)
      acd142(42)=acd142(24)*acd142(30)
      acd142(43)=acd142(10)*acd142(9)
      acd142(44)=acd142(16)*acd142(29)
      acd142(45)=acd142(14)*acd142(28)
      acd142(42)=acd142(45)+acd142(44)+acd142(42)+acd142(43)
      acd142(42)=acd142(13)*acd142(42)
      acd142(43)=acd142(27)*acd142(30)
      acd142(44)=acd142(3)*acd142(9)
      acd142(45)=acd142(19)*acd142(29)
      acd142(46)=acd142(18)*acd142(28)
      acd142(43)=acd142(46)+acd142(45)+acd142(43)+acd142(44)
      acd142(43)=acd142(8)*acd142(43)
      acd142(44)=acd142(31)*acd142(32)
      acd142(45)=acd142(33)*acd142(34)
      acd142(46)=acd142(4)*acd142(15)
      acd142(44)=acd142(46)+acd142(44)+acd142(45)
      acd142(44)=acd142(18)*acd142(44)
      acd142(45)=acd142(32)*acd142(35)
      acd142(46)=acd142(36)*acd142(34)
      acd142(47)=acd142(11)*acd142(15)
      acd142(45)=acd142(47)+acd142(45)+acd142(46)
      acd142(45)=acd142(14)*acd142(45)
      acd142(46)=acd142(27)*acd142(33)
      acd142(47)=acd142(24)*acd142(36)
      acd142(46)=acd142(47)+acd142(46)
      acd142(46)=acd142(38)*acd142(46)
      acd142(47)=acd142(33)*acd142(37)
      acd142(48)=acd142(4)*acd142(17)
      acd142(47)=acd142(47)+acd142(48)
      acd142(47)=acd142(19)*acd142(47)
      acd142(48)=acd142(36)*acd142(37)
      acd142(49)=acd142(11)*acd142(17)
      acd142(48)=acd142(48)+acd142(49)
      acd142(48)=acd142(16)*acd142(48)
      acd142(49)=acd142(1)*acd142(2)
      brack=acd142(39)+acd142(40)+acd142(41)+acd142(42)+acd142(43)+acd142(44)+a&
      &cd142(45)+acd142(46)+acd142(47)+acd142(48)+2.0_ki*acd142(49)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd142h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd142
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd142h4
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k3-k5
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d142h4l1d
