module     p2_gg_httbar_d172h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d172h4l1d.f90
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
      use p2_gg_httbar_abbrevd172h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(46) :: acd172
      complex(ki) :: brack
      acd172(1)=dotproduct(qshift,qshift)
      acd172(2)=abb172(20)
      acd172(3)=dotproduct(qshift,spvak2e1)
      acd172(4)=dotproduct(qshift,spvae1k2)
      acd172(5)=abb172(19)
      acd172(6)=dotproduct(qshift,spvae1l4)
      acd172(7)=abb172(17)
      acd172(8)=dotproduct(qshift,spvae1l5)
      acd172(9)=abb172(24)
      acd172(10)=dotproduct(qshift,spvae1e2)
      acd172(11)=abb172(23)
      acd172(12)=abb172(14)
      acd172(13)=dotproduct(qshift,spval4e1)
      acd172(14)=abb172(27)
      acd172(15)=dotproduct(qshift,spval5e1)
      acd172(16)=abb172(15)
      acd172(17)=dotproduct(qshift,spvae2e1)
      acd172(18)=abb172(21)
      acd172(19)=abb172(12)
      acd172(20)=abb172(80)
      acd172(21)=abb172(66)
      acd172(22)=abb172(70)
      acd172(23)=abb172(71)
      acd172(24)=abb172(87)
      acd172(25)=abb172(93)
      acd172(26)=abb172(30)
      acd172(27)=abb172(59)
      acd172(28)=abb172(69)
      acd172(29)=abb172(62)
      acd172(30)=abb172(52)
      acd172(31)=dotproduct(qshift,spval3e1)
      acd172(32)=abb172(18)
      acd172(33)=dotproduct(qshift,spvae1l3)
      acd172(34)=abb172(16)
      acd172(35)=abb172(13)
      acd172(36)=acd172(8)*acd172(9)
      acd172(37)=acd172(10)*acd172(11)
      acd172(38)=acd172(6)*acd172(7)
      acd172(39)=acd172(4)*acd172(5)
      acd172(36)=acd172(39)+acd172(38)+acd172(37)-acd172(12)+acd172(36)
      acd172(36)=acd172(3)*acd172(36)
      acd172(37)=-acd172(13)*acd172(20)
      acd172(38)=acd172(17)*acd172(22)
      acd172(39)=acd172(15)*acd172(21)
      acd172(37)=acd172(39)+acd172(38)-acd172(23)+acd172(37)
      acd172(37)=acd172(6)*acd172(37)
      acd172(38)=acd172(13)*acd172(14)
      acd172(39)=acd172(17)*acd172(18)
      acd172(40)=acd172(15)*acd172(16)
      acd172(38)=acd172(40)+acd172(39)-acd172(19)+acd172(38)
      acd172(38)=acd172(4)*acd172(38)
      acd172(39)=-acd172(8)*acd172(20)
      acd172(40)=acd172(10)*acd172(25)
      acd172(39)=acd172(40)-acd172(29)+acd172(39)
      acd172(39)=acd172(15)*acd172(39)
      acd172(40)=-acd172(33)*acd172(34)
      acd172(41)=-acd172(31)*acd172(32)
      acd172(42)=acd172(1)*acd172(2)
      acd172(43)=-acd172(13)*acd172(28)
      acd172(44)=-acd172(8)*acd172(24)
      acd172(45)=-acd172(17)*acd172(30)
      acd172(46)=acd172(17)*acd172(26)
      acd172(46)=-acd172(27)+acd172(46)
      acd172(46)=acd172(10)*acd172(46)
      brack=acd172(35)+acd172(36)+acd172(37)+acd172(38)+acd172(39)+acd172(40)+a&
      &cd172(41)+acd172(42)+acd172(43)+acd172(44)+acd172(45)+acd172(46)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd172h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(57) :: acd172
      complex(ki) :: brack
      acd172(1)=qshift(iv1)
      acd172(2)=abb172(20)
      acd172(3)=spvak2e1(iv1)
      acd172(4)=dotproduct(qshift,spvae1k2)
      acd172(5)=abb172(19)
      acd172(6)=dotproduct(qshift,spvae1l4)
      acd172(7)=abb172(17)
      acd172(8)=dotproduct(qshift,spvae1l5)
      acd172(9)=abb172(24)
      acd172(10)=dotproduct(qshift,spvae1e2)
      acd172(11)=abb172(23)
      acd172(12)=abb172(14)
      acd172(13)=spvae1k2(iv1)
      acd172(14)=dotproduct(qshift,spvak2e1)
      acd172(15)=dotproduct(qshift,spval4e1)
      acd172(16)=abb172(27)
      acd172(17)=dotproduct(qshift,spval5e1)
      acd172(18)=abb172(15)
      acd172(19)=dotproduct(qshift,spvae2e1)
      acd172(20)=abb172(21)
      acd172(21)=abb172(12)
      acd172(22)=spvae1l4(iv1)
      acd172(23)=abb172(80)
      acd172(24)=abb172(66)
      acd172(25)=abb172(70)
      acd172(26)=abb172(71)
      acd172(27)=spvae1l5(iv1)
      acd172(28)=abb172(87)
      acd172(29)=spvae1e2(iv1)
      acd172(30)=abb172(93)
      acd172(31)=abb172(30)
      acd172(32)=abb172(59)
      acd172(33)=spval4e1(iv1)
      acd172(34)=abb172(69)
      acd172(35)=spval5e1(iv1)
      acd172(36)=abb172(62)
      acd172(37)=spvae2e1(iv1)
      acd172(38)=abb172(52)
      acd172(39)=spval3e1(iv1)
      acd172(40)=abb172(18)
      acd172(41)=spvae1l3(iv1)
      acd172(42)=abb172(16)
      acd172(43)=-acd172(10)*acd172(30)
      acd172(44)=acd172(23)*acd172(8)
      acd172(45)=-acd172(6)*acd172(24)
      acd172(46)=-acd172(4)*acd172(18)
      acd172(43)=acd172(46)+acd172(45)+acd172(44)+acd172(36)+acd172(43)
      acd172(43)=acd172(35)*acd172(43)
      acd172(44)=-acd172(19)*acd172(25)
      acd172(45)=acd172(23)*acd172(15)
      acd172(46)=-acd172(17)*acd172(24)
      acd172(47)=-acd172(14)*acd172(7)
      acd172(44)=acd172(47)+acd172(46)+acd172(45)+acd172(26)+acd172(44)
      acd172(44)=acd172(22)*acd172(44)
      acd172(45)=-acd172(15)*acd172(16)
      acd172(46)=-acd172(19)*acd172(20)
      acd172(47)=-acd172(17)*acd172(18)
      acd172(48)=-acd172(14)*acd172(5)
      acd172(45)=acd172(48)+acd172(47)+acd172(46)+acd172(21)+acd172(45)
      acd172(45)=acd172(13)*acd172(45)
      acd172(46)=-acd172(8)*acd172(9)
      acd172(47)=-acd172(10)*acd172(11)
      acd172(48)=-acd172(6)*acd172(7)
      acd172(49)=-acd172(4)*acd172(5)
      acd172(46)=acd172(49)+acd172(48)+acd172(47)+acd172(12)+acd172(46)
      acd172(46)=acd172(3)*acd172(46)
      acd172(47)=-acd172(29)*acd172(30)
      acd172(48)=acd172(23)*acd172(27)
      acd172(47)=acd172(47)+acd172(48)
      acd172(47)=acd172(17)*acd172(47)
      acd172(48)=-acd172(27)*acd172(9)
      acd172(49)=-acd172(29)*acd172(11)
      acd172(48)=acd172(48)+acd172(49)
      acd172(48)=acd172(14)*acd172(48)
      acd172(49)=-acd172(37)*acd172(25)
      acd172(50)=acd172(23)*acd172(33)
      acd172(49)=acd172(49)+acd172(50)
      acd172(49)=acd172(6)*acd172(49)
      acd172(50)=-acd172(33)*acd172(16)
      acd172(51)=-acd172(37)*acd172(20)
      acd172(50)=acd172(50)+acd172(51)
      acd172(50)=acd172(4)*acd172(50)
      acd172(51)=acd172(41)*acd172(42)
      acd172(52)=acd172(39)*acd172(40)
      acd172(53)=acd172(1)*acd172(2)
      acd172(54)=acd172(33)*acd172(34)
      acd172(55)=acd172(27)*acd172(28)
      acd172(56)=-acd172(10)*acd172(31)
      acd172(56)=acd172(38)+acd172(56)
      acd172(56)=acd172(37)*acd172(56)
      acd172(57)=-acd172(19)*acd172(31)
      acd172(57)=acd172(32)+acd172(57)
      acd172(57)=acd172(29)*acd172(57)
      brack=acd172(43)+acd172(44)+acd172(45)+acd172(46)+acd172(47)+acd172(48)+a&
      &cd172(49)+acd172(50)+acd172(51)+acd172(52)-2.0_ki*acd172(53)+acd172(54)+&
      &acd172(55)+acd172(56)+acd172(57)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd172h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd172
      complex(ki) :: brack
      acd172(1)=d(iv1,iv2)
      acd172(2)=abb172(20)
      acd172(3)=spvak2e1(iv1)
      acd172(4)=spvae1k2(iv2)
      acd172(5)=abb172(19)
      acd172(6)=spvae1l4(iv2)
      acd172(7)=abb172(17)
      acd172(8)=spvae1l5(iv2)
      acd172(9)=abb172(24)
      acd172(10)=spvae1e2(iv2)
      acd172(11)=abb172(23)
      acd172(12)=spvak2e1(iv2)
      acd172(13)=spvae1k2(iv1)
      acd172(14)=spvae1l4(iv1)
      acd172(15)=spvae1l5(iv1)
      acd172(16)=spvae1e2(iv1)
      acd172(17)=spval4e1(iv2)
      acd172(18)=abb172(27)
      acd172(19)=spval5e1(iv2)
      acd172(20)=abb172(15)
      acd172(21)=spvae2e1(iv2)
      acd172(22)=abb172(21)
      acd172(23)=spval4e1(iv1)
      acd172(24)=spval5e1(iv1)
      acd172(25)=spvae2e1(iv1)
      acd172(26)=abb172(80)
      acd172(27)=abb172(66)
      acd172(28)=abb172(70)
      acd172(29)=abb172(93)
      acd172(30)=abb172(30)
      acd172(31)=acd172(9)*acd172(15)
      acd172(32)=acd172(16)*acd172(11)
      acd172(33)=acd172(14)*acd172(7)
      acd172(34)=acd172(13)*acd172(5)
      acd172(31)=acd172(34)+acd172(33)+acd172(31)+acd172(32)
      acd172(31)=acd172(12)*acd172(31)
      acd172(32)=acd172(8)*acd172(9)
      acd172(33)=acd172(10)*acd172(11)
      acd172(34)=acd172(6)*acd172(7)
      acd172(35)=acd172(4)*acd172(5)
      acd172(32)=acd172(35)+acd172(34)+acd172(32)+acd172(33)
      acd172(32)=acd172(3)*acd172(32)
      acd172(33)=acd172(21)*acd172(28)
      acd172(34)=-acd172(26)*acd172(17)
      acd172(35)=acd172(19)*acd172(27)
      acd172(33)=acd172(35)+acd172(33)+acd172(34)
      acd172(33)=acd172(14)*acd172(33)
      acd172(34)=acd172(17)*acd172(18)
      acd172(35)=acd172(21)*acd172(22)
      acd172(36)=acd172(19)*acd172(20)
      acd172(34)=acd172(36)+acd172(34)+acd172(35)
      acd172(34)=acd172(13)*acd172(34)
      acd172(35)=acd172(25)*acd172(28)
      acd172(36)=-acd172(26)*acd172(23)
      acd172(37)=acd172(24)*acd172(27)
      acd172(35)=acd172(37)+acd172(35)+acd172(36)
      acd172(35)=acd172(6)*acd172(35)
      acd172(36)=acd172(18)*acd172(23)
      acd172(37)=acd172(25)*acd172(22)
      acd172(38)=acd172(24)*acd172(20)
      acd172(36)=acd172(38)+acd172(36)+acd172(37)
      acd172(36)=acd172(4)*acd172(36)
      acd172(37)=acd172(16)*acd172(21)
      acd172(38)=acd172(10)*acd172(25)
      acd172(37)=acd172(38)+acd172(37)
      acd172(37)=acd172(30)*acd172(37)
      acd172(38)=acd172(10)*acd172(29)
      acd172(39)=-acd172(26)*acd172(8)
      acd172(38)=acd172(38)+acd172(39)
      acd172(38)=acd172(24)*acd172(38)
      acd172(39)=acd172(16)*acd172(29)
      acd172(40)=-acd172(26)*acd172(15)
      acd172(39)=acd172(39)+acd172(40)
      acd172(39)=acd172(19)*acd172(39)
      acd172(40)=acd172(1)*acd172(2)
      brack=acd172(31)+acd172(32)+acd172(33)+acd172(34)+acd172(35)+acd172(36)+a&
      &cd172(37)+acd172(38)+acd172(39)+2.0_ki*acd172(40)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd172h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd172
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd172h4
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
end module     p2_gg_httbar_d172h4l1d
