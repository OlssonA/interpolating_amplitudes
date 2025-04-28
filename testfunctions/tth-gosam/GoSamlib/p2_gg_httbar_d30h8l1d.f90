module     p2_gg_httbar_d30h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d30h8l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd30h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd30
      complex(ki) :: brack
      acd30(1)=dotproduct(qshift,spvak1e2)
      acd30(2)=dotproduct(qshift,spvae2k2)
      acd30(3)=abb30(20)
      acd30(4)=dotproduct(qshift,spvae2l3)
      acd30(5)=abb30(16)
      acd30(6)=abb30(15)
      acd30(7)=dotproduct(qshift,spval3e2)
      acd30(8)=abb30(13)
      acd30(9)=dotproduct(qshift,spval4e2)
      acd30(10)=abb30(18)
      acd30(11)=dotproduct(qshift,spval5e2)
      acd30(12)=abb30(22)
      acd30(13)=dotproduct(qshift,spvae1e2)
      acd30(14)=abb30(19)
      acd30(15)=abb30(10)
      acd30(16)=abb30(35)
      acd30(17)=abb30(33)
      acd30(18)=abb30(21)
      acd30(19)=dotproduct(qshift,spvae2k1)
      acd30(20)=abb30(63)
      acd30(21)=abb30(42)
      acd30(22)=abb30(14)
      acd30(23)=dotproduct(qshift,spvae2l5)
      acd30(24)=abb30(61)
      acd30(25)=dotproduct(qshift,spvae2e1)
      acd30(26)=abb30(38)
      acd30(27)=abb30(34)
      acd30(28)=abb30(40)
      acd30(29)=abb30(41)
      acd30(30)=abb30(39)
      acd30(31)=abb30(23)
      acd30(32)=abb30(17)
      acd30(33)=abb30(26)
      acd30(34)=abb30(12)
      acd30(35)=abb30(11)
      acd30(36)=acd30(3)*acd30(1)
      acd30(37)=acd30(8)*acd30(7)
      acd30(38)=acd30(10)*acd30(9)
      acd30(39)=acd30(12)*acd30(11)
      acd30(40)=acd30(14)*acd30(13)
      acd30(36)=-acd30(15)+acd30(40)+acd30(39)+acd30(38)+acd30(37)+acd30(36)
      acd30(36)=acd30(2)*acd30(36)
      acd30(37)=acd30(20)*acd30(19)
      acd30(38)=-acd30(24)*acd30(23)
      acd30(39)=acd30(26)*acd30(25)
      acd30(37)=-acd30(27)+acd30(39)+acd30(38)+acd30(37)
      acd30(37)=acd30(7)*acd30(37)
      acd30(38)=acd30(21)*acd30(19)
      acd30(39)=-acd30(28)*acd30(23)
      acd30(40)=acd30(29)*acd30(25)
      acd30(38)=-acd30(30)+acd30(40)+acd30(39)+acd30(38)
      acd30(38)=acd30(9)*acd30(38)
      acd30(39)=acd30(5)*acd30(1)
      acd30(40)=acd30(16)*acd30(11)
      acd30(41)=acd30(17)*acd30(13)
      acd30(39)=-acd30(18)+acd30(41)+acd30(40)+acd30(39)
      acd30(39)=acd30(4)*acd30(39)
      acd30(40)=-acd30(6)*acd30(1)
      acd30(41)=-acd30(22)*acd30(19)
      acd30(42)=-acd30(31)*acd30(11)
      acd30(43)=-acd30(32)*acd30(13)
      acd30(44)=-acd30(33)*acd30(23)
      acd30(45)=-acd30(34)*acd30(25)
      brack=acd30(35)+acd30(36)+acd30(37)+acd30(38)+acd30(39)+acd30(40)+acd30(4&
      &1)+acd30(42)+acd30(43)+acd30(44)+acd30(45)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd30h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(58) :: acd30
      complex(ki) :: brack
      acd30(1)=spvak1e2(iv1)
      acd30(2)=dotproduct(qshift,spvae2k2)
      acd30(3)=abb30(20)
      acd30(4)=dotproduct(qshift,spvae2l3)
      acd30(5)=abb30(16)
      acd30(6)=abb30(15)
      acd30(7)=spvae2k2(iv1)
      acd30(8)=dotproduct(qshift,spvak1e2)
      acd30(9)=dotproduct(qshift,spval3e2)
      acd30(10)=abb30(13)
      acd30(11)=dotproduct(qshift,spval4e2)
      acd30(12)=abb30(18)
      acd30(13)=dotproduct(qshift,spval5e2)
      acd30(14)=abb30(22)
      acd30(15)=dotproduct(qshift,spvae1e2)
      acd30(16)=abb30(19)
      acd30(17)=abb30(10)
      acd30(18)=spvae2l3(iv1)
      acd30(19)=abb30(35)
      acd30(20)=abb30(33)
      acd30(21)=abb30(21)
      acd30(22)=spvae2k1(iv1)
      acd30(23)=abb30(63)
      acd30(24)=abb30(42)
      acd30(25)=abb30(14)
      acd30(26)=spval3e2(iv1)
      acd30(27)=dotproduct(qshift,spvae2k1)
      acd30(28)=dotproduct(qshift,spvae2l5)
      acd30(29)=abb30(61)
      acd30(30)=dotproduct(qshift,spvae2e1)
      acd30(31)=abb30(38)
      acd30(32)=abb30(34)
      acd30(33)=spval4e2(iv1)
      acd30(34)=abb30(40)
      acd30(35)=abb30(41)
      acd30(36)=abb30(39)
      acd30(37)=spval5e2(iv1)
      acd30(38)=abb30(23)
      acd30(39)=spvae1e2(iv1)
      acd30(40)=abb30(17)
      acd30(41)=spvae2l5(iv1)
      acd30(42)=abb30(26)
      acd30(43)=spvae2e1(iv1)
      acd30(44)=abb30(12)
      acd30(45)=-acd30(39)*acd30(16)
      acd30(46)=-acd30(37)*acd30(14)
      acd30(47)=-acd30(1)*acd30(3)
      acd30(48)=-acd30(33)*acd30(12)
      acd30(49)=-acd30(26)*acd30(10)
      acd30(45)=acd30(49)+acd30(48)+acd30(47)+acd30(45)+acd30(46)
      acd30(45)=acd30(2)*acd30(45)
      acd30(46)=-acd30(15)*acd30(16)
      acd30(47)=-acd30(13)*acd30(14)
      acd30(48)=-acd30(3)*acd30(8)
      acd30(49)=-acd30(11)*acd30(12)
      acd30(50)=-acd30(9)*acd30(10)
      acd30(46)=acd30(50)+acd30(49)+acd30(48)+acd30(47)+acd30(17)+acd30(46)
      acd30(46)=acd30(7)*acd30(46)
      acd30(47)=-acd30(15)*acd30(20)
      acd30(48)=-acd30(13)*acd30(19)
      acd30(49)=-acd30(5)*acd30(8)
      acd30(47)=acd30(49)+acd30(48)+acd30(21)+acd30(47)
      acd30(47)=acd30(18)*acd30(47)
      acd30(48)=-acd30(43)*acd30(35)
      acd30(49)=acd30(41)*acd30(34)
      acd30(50)=-acd30(22)*acd30(24)
      acd30(48)=acd30(50)+acd30(48)+acd30(49)
      acd30(48)=acd30(11)*acd30(48)
      acd30(49)=-acd30(43)*acd30(31)
      acd30(50)=acd30(41)*acd30(29)
      acd30(51)=-acd30(22)*acd30(23)
      acd30(49)=acd30(51)+acd30(49)+acd30(50)
      acd30(49)=acd30(9)*acd30(49)
      acd30(50)=-acd30(30)*acd30(35)
      acd30(51)=acd30(28)*acd30(34)
      acd30(52)=-acd30(24)*acd30(27)
      acd30(50)=acd30(52)+acd30(51)+acd30(36)+acd30(50)
      acd30(50)=acd30(33)*acd30(50)
      acd30(51)=-acd30(30)*acd30(31)
      acd30(52)=acd30(28)*acd30(29)
      acd30(53)=-acd30(23)*acd30(27)
      acd30(51)=acd30(53)+acd30(52)+acd30(32)+acd30(51)
      acd30(51)=acd30(26)*acd30(51)
      acd30(52)=-acd30(39)*acd30(20)
      acd30(53)=-acd30(37)*acd30(19)
      acd30(52)=acd30(52)+acd30(53)
      acd30(52)=acd30(4)*acd30(52)
      acd30(53)=acd30(43)*acd30(44)
      acd30(54)=acd30(41)*acd30(42)
      acd30(55)=acd30(39)*acd30(40)
      acd30(56)=acd30(37)*acd30(38)
      acd30(57)=acd30(22)*acd30(25)
      acd30(58)=-acd30(4)*acd30(5)
      acd30(58)=acd30(6)+acd30(58)
      acd30(58)=acd30(1)*acd30(58)
      brack=acd30(45)+acd30(46)+acd30(47)+acd30(48)+acd30(49)+acd30(50)+acd30(5&
      &1)+acd30(52)+acd30(53)+acd30(54)+acd30(55)+acd30(56)+acd30(57)+acd30(58)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd30h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(44) :: acd30
      complex(ki) :: brack
      acd30(1)=spvak1e2(iv1)
      acd30(2)=spvae2k2(iv2)
      acd30(3)=abb30(20)
      acd30(4)=spvae2l3(iv2)
      acd30(5)=abb30(16)
      acd30(6)=spvak1e2(iv2)
      acd30(7)=spvae2k2(iv1)
      acd30(8)=spvae2l3(iv1)
      acd30(9)=spval3e2(iv2)
      acd30(10)=abb30(13)
      acd30(11)=spval4e2(iv2)
      acd30(12)=abb30(18)
      acd30(13)=spval5e2(iv2)
      acd30(14)=abb30(22)
      acd30(15)=spvae1e2(iv2)
      acd30(16)=abb30(19)
      acd30(17)=spval3e2(iv1)
      acd30(18)=spval4e2(iv1)
      acd30(19)=spval5e2(iv1)
      acd30(20)=spvae1e2(iv1)
      acd30(21)=abb30(35)
      acd30(22)=abb30(33)
      acd30(23)=spvae2k1(iv1)
      acd30(24)=abb30(63)
      acd30(25)=abb30(42)
      acd30(26)=spvae2k1(iv2)
      acd30(27)=spvae2l5(iv2)
      acd30(28)=abb30(61)
      acd30(29)=spvae2e1(iv2)
      acd30(30)=abb30(38)
      acd30(31)=spvae2l5(iv1)
      acd30(32)=spvae2e1(iv1)
      acd30(33)=abb30(40)
      acd30(34)=abb30(41)
      acd30(35)=acd30(15)*acd30(16)
      acd30(36)=acd30(13)*acd30(14)
      acd30(37)=acd30(3)*acd30(6)
      acd30(38)=acd30(11)*acd30(12)
      acd30(39)=acd30(9)*acd30(10)
      acd30(35)=acd30(39)+acd30(38)+acd30(37)+acd30(35)+acd30(36)
      acd30(35)=acd30(7)*acd30(35)
      acd30(36)=acd30(16)*acd30(20)
      acd30(37)=acd30(14)*acd30(19)
      acd30(38)=acd30(1)*acd30(3)
      acd30(39)=acd30(18)*acd30(12)
      acd30(40)=acd30(17)*acd30(10)
      acd30(36)=acd30(40)+acd30(39)+acd30(38)+acd30(36)+acd30(37)
      acd30(36)=acd30(2)*acd30(36)
      acd30(37)=acd30(15)*acd30(22)
      acd30(38)=acd30(13)*acd30(21)
      acd30(39)=acd30(5)*acd30(6)
      acd30(37)=acd30(39)+acd30(37)+acd30(38)
      acd30(37)=acd30(8)*acd30(37)
      acd30(38)=acd30(20)*acd30(22)
      acd30(39)=acd30(19)*acd30(21)
      acd30(40)=acd30(1)*acd30(5)
      acd30(38)=acd30(40)+acd30(38)+acd30(39)
      acd30(38)=acd30(4)*acd30(38)
      acd30(39)=acd30(29)*acd30(34)
      acd30(40)=-acd30(27)*acd30(33)
      acd30(41)=acd30(25)*acd30(26)
      acd30(39)=acd30(41)+acd30(39)+acd30(40)
      acd30(39)=acd30(18)*acd30(39)
      acd30(40)=acd30(29)*acd30(30)
      acd30(41)=-acd30(27)*acd30(28)
      acd30(42)=acd30(24)*acd30(26)
      acd30(40)=acd30(42)+acd30(40)+acd30(41)
      acd30(40)=acd30(17)*acd30(40)
      acd30(41)=acd30(32)*acd30(34)
      acd30(42)=-acd30(31)*acd30(33)
      acd30(43)=acd30(23)*acd30(25)
      acd30(41)=acd30(43)+acd30(41)+acd30(42)
      acd30(41)=acd30(11)*acd30(41)
      acd30(42)=acd30(30)*acd30(32)
      acd30(43)=-acd30(28)*acd30(31)
      acd30(44)=acd30(23)*acd30(24)
      acd30(42)=acd30(44)+acd30(42)+acd30(43)
      acd30(42)=acd30(9)*acd30(42)
      brack=acd30(35)+acd30(36)+acd30(37)+acd30(38)+acd30(39)+acd30(40)+acd30(4&
      &1)+acd30(42)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd30h8
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = -k2+k3+k4
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
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d30h8l1d
