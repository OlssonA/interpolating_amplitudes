module     p2_gg_httbar_d76h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d76h8l1d.f90
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
      use p2_gg_httbar_abbrevd76h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(49) :: acd76
      complex(ki) :: brack
      acd76(1)=dotproduct(qshift,qshift)
      acd76(2)=dotproduct(qshift,spvak2e1)
      acd76(3)=abb76(28)
      acd76(4)=dotproduct(qshift,spval4e1)
      acd76(5)=abb76(20)
      acd76(6)=dotproduct(qshift,spvae1l4)
      acd76(7)=abb76(25)
      acd76(8)=dotproduct(qshift,spvae1l5)
      acd76(9)=abb76(24)
      acd76(10)=dotproduct(qshift,spvae1e2)
      acd76(11)=abb76(22)
      acd76(12)=dotproduct(qshift,spvae2e1)
      acd76(13)=abb76(13)
      acd76(14)=abb76(16)
      acd76(15)=abb76(10)
      acd76(16)=abb76(12)
      acd76(17)=abb76(15)
      acd76(18)=dotproduct(qshift,spvae1k2)
      acd76(19)=abb76(29)
      acd76(20)=dotproduct(qshift,spvae1l3)
      acd76(21)=abb76(27)
      acd76(22)=abb76(18)
      acd76(23)=abb76(34)
      acd76(24)=abb76(35)
      acd76(25)=abb76(33)
      acd76(26)=dotproduct(qshift,spval3e1)
      acd76(27)=abb76(41)
      acd76(28)=abb76(11)
      acd76(29)=abb76(26)
      acd76(30)=abb76(38)
      acd76(31)=abb76(14)
      acd76(32)=abb76(40)
      acd76(33)=abb76(17)
      acd76(34)=abb76(19)
      acd76(35)=abb76(23)
      acd76(36)=abb76(42)
      acd76(37)=abb76(21)
      acd76(38)=abb76(39)
      acd76(39)=abb76(9)
      acd76(40)=-acd76(3)*acd76(1)
      acd76(41)=acd76(15)*acd76(6)
      acd76(42)=acd76(16)*acd76(8)
      acd76(43)=acd76(17)*acd76(10)
      acd76(44)=acd76(19)*acd76(18)
      acd76(45)=acd76(21)*acd76(20)
      acd76(40)=-acd76(22)+acd76(45)+acd76(44)+acd76(43)+acd76(42)+acd76(41)+ac&
      &d76(40)
      acd76(40)=acd76(2)*acd76(40)
      acd76(41)=acd76(5)*acd76(4)
      acd76(42)=acd76(7)*acd76(6)
      acd76(43)=-acd76(9)*acd76(8)
      acd76(44)=acd76(11)*acd76(10)
      acd76(45)=-acd76(13)*acd76(12)
      acd76(41)=acd76(14)+acd76(45)+acd76(44)+acd76(43)+acd76(42)+acd76(41)
      acd76(41)=acd76(1)*acd76(41)
      acd76(42)=acd76(27)*acd76(6)
      acd76(43)=-acd76(30)*acd76(8)
      acd76(44)=acd76(32)*acd76(10)
      acd76(42)=-acd76(38)+acd76(44)+acd76(43)+acd76(42)
      acd76(42)=acd76(26)*acd76(42)
      acd76(43)=acd76(23)*acd76(4)
      acd76(44)=acd76(29)*acd76(12)
      acd76(43)=-acd76(31)+acd76(44)+acd76(43)
      acd76(43)=acd76(8)*acd76(43)
      acd76(44)=acd76(24)*acd76(4)
      acd76(45)=-acd76(34)*acd76(12)
      acd76(44)=-acd76(37)+acd76(45)+acd76(44)
      acd76(44)=acd76(20)*acd76(44)
      acd76(45)=-acd76(25)*acd76(4)
      acd76(46)=-acd76(28)*acd76(6)
      acd76(47)=-acd76(33)*acd76(10)
      acd76(48)=-acd76(35)*acd76(12)
      acd76(49)=-acd76(36)*acd76(18)
      brack=acd76(39)+acd76(40)+acd76(41)+acd76(42)+acd76(43)+acd76(44)+acd76(4&
      &5)+acd76(46)+acd76(47)+acd76(48)+acd76(49)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(61) :: acd76
      complex(ki) :: brack
      acd76(1)=qshift(iv1)
      acd76(2)=dotproduct(qshift,spvak2e1)
      acd76(3)=abb76(28)
      acd76(4)=dotproduct(qshift,spval4e1)
      acd76(5)=abb76(20)
      acd76(6)=dotproduct(qshift,spvae1l4)
      acd76(7)=abb76(25)
      acd76(8)=dotproduct(qshift,spvae1l5)
      acd76(9)=abb76(24)
      acd76(10)=dotproduct(qshift,spvae1e2)
      acd76(11)=abb76(22)
      acd76(12)=dotproduct(qshift,spvae2e1)
      acd76(13)=abb76(13)
      acd76(14)=abb76(16)
      acd76(15)=spvak2e1(iv1)
      acd76(16)=dotproduct(qshift,qshift)
      acd76(17)=abb76(10)
      acd76(18)=abb76(12)
      acd76(19)=abb76(15)
      acd76(20)=dotproduct(qshift,spvae1k2)
      acd76(21)=abb76(29)
      acd76(22)=dotproduct(qshift,spvae1l3)
      acd76(23)=abb76(27)
      acd76(24)=abb76(18)
      acd76(25)=spval4e1(iv1)
      acd76(26)=abb76(34)
      acd76(27)=abb76(35)
      acd76(28)=abb76(33)
      acd76(29)=spvae1l4(iv1)
      acd76(30)=dotproduct(qshift,spval3e1)
      acd76(31)=abb76(41)
      acd76(32)=abb76(11)
      acd76(33)=spvae1l5(iv1)
      acd76(34)=abb76(26)
      acd76(35)=abb76(38)
      acd76(36)=abb76(14)
      acd76(37)=spvae1e2(iv1)
      acd76(38)=abb76(40)
      acd76(39)=abb76(17)
      acd76(40)=spvae2e1(iv1)
      acd76(41)=abb76(19)
      acd76(42)=abb76(23)
      acd76(43)=spvae1k2(iv1)
      acd76(44)=abb76(42)
      acd76(45)=spvae1l3(iv1)
      acd76(46)=abb76(21)
      acd76(47)=spval3e1(iv1)
      acd76(48)=abb76(39)
      acd76(49)=acd76(21)*acd76(20)
      acd76(50)=acd76(22)*acd76(23)
      acd76(51)=acd76(10)*acd76(19)
      acd76(52)=acd76(6)*acd76(17)
      acd76(53)=acd76(8)*acd76(18)
      acd76(54)=-acd76(16)*acd76(3)
      acd76(49)=acd76(54)+acd76(53)+acd76(52)+acd76(51)+acd76(50)-acd76(24)+acd&
      &76(49)
      acd76(49)=acd76(15)*acd76(49)
      acd76(50)=-acd76(12)*acd76(13)
      acd76(51)=acd76(10)*acd76(11)
      acd76(52)=acd76(6)*acd76(7)
      acd76(53)=acd76(4)*acd76(5)
      acd76(54)=-acd76(8)*acd76(9)
      acd76(55)=-acd76(2)*acd76(3)
      acd76(50)=acd76(55)+acd76(54)+acd76(53)+acd76(52)+acd76(51)+acd76(14)+acd&
      &76(50)
      acd76(50)=acd76(1)*acd76(50)
      acd76(51)=-acd76(40)*acd76(13)
      acd76(52)=acd76(37)*acd76(11)
      acd76(53)=acd76(29)*acd76(7)
      acd76(54)=acd76(25)*acd76(5)
      acd76(55)=-acd76(33)*acd76(9)
      acd76(51)=acd76(55)+acd76(54)+acd76(53)+acd76(51)+acd76(52)
      acd76(51)=acd76(16)*acd76(51)
      acd76(52)=acd76(21)*acd76(43)
      acd76(53)=acd76(45)*acd76(23)
      acd76(54)=acd76(37)*acd76(19)
      acd76(55)=acd76(29)*acd76(17)
      acd76(56)=acd76(33)*acd76(18)
      acd76(52)=acd76(56)+acd76(55)+acd76(54)+acd76(52)+acd76(53)
      acd76(52)=acd76(2)*acd76(52)
      acd76(53)=-acd76(47)*acd76(35)
      acd76(54)=acd76(40)*acd76(34)
      acd76(55)=acd76(25)*acd76(26)
      acd76(53)=acd76(55)+acd76(53)+acd76(54)
      acd76(53)=acd76(8)*acd76(53)
      acd76(54)=-acd76(30)*acd76(35)
      acd76(55)=acd76(12)*acd76(34)
      acd76(56)=acd76(4)*acd76(26)
      acd76(54)=acd76(56)+acd76(55)-acd76(36)+acd76(54)
      acd76(54)=acd76(33)*acd76(54)
      acd76(55)=acd76(10)*acd76(38)
      acd76(56)=acd76(6)*acd76(31)
      acd76(55)=acd76(56)-acd76(48)+acd76(55)
      acd76(55)=acd76(47)*acd76(55)
      acd76(56)=-acd76(12)*acd76(41)
      acd76(57)=acd76(4)*acd76(27)
      acd76(56)=acd76(57)-acd76(46)+acd76(56)
      acd76(56)=acd76(45)*acd76(56)
      acd76(57)=-acd76(43)*acd76(44)
      acd76(58)=-acd76(22)*acd76(41)
      acd76(58)=-acd76(42)+acd76(58)
      acd76(58)=acd76(40)*acd76(58)
      acd76(59)=acd76(30)*acd76(38)
      acd76(59)=-acd76(39)+acd76(59)
      acd76(59)=acd76(37)*acd76(59)
      acd76(60)=acd76(30)*acd76(31)
      acd76(60)=-acd76(32)+acd76(60)
      acd76(60)=acd76(29)*acd76(60)
      acd76(61)=acd76(22)*acd76(27)
      acd76(61)=-acd76(28)+acd76(61)
      acd76(61)=acd76(25)*acd76(61)
      brack=acd76(49)+2.0_ki*acd76(50)+acd76(51)+acd76(52)+acd76(53)+acd76(54)+&
      &acd76(55)+acd76(56)+acd76(57)+acd76(58)+acd76(59)+acd76(60)+acd76(61)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(56) :: acd76
      complex(ki) :: brack
      acd76(1)=d(iv1,iv2)
      acd76(2)=dotproduct(qshift,spvak2e1)
      acd76(3)=abb76(28)
      acd76(4)=dotproduct(qshift,spval4e1)
      acd76(5)=abb76(20)
      acd76(6)=dotproduct(qshift,spvae1l4)
      acd76(7)=abb76(25)
      acd76(8)=dotproduct(qshift,spvae1l5)
      acd76(9)=abb76(24)
      acd76(10)=dotproduct(qshift,spvae1e2)
      acd76(11)=abb76(22)
      acd76(12)=dotproduct(qshift,spvae2e1)
      acd76(13)=abb76(13)
      acd76(14)=abb76(16)
      acd76(15)=qshift(iv1)
      acd76(16)=spvak2e1(iv2)
      acd76(17)=spval4e1(iv2)
      acd76(18)=spvae1l4(iv2)
      acd76(19)=spvae1l5(iv2)
      acd76(20)=spvae1e2(iv2)
      acd76(21)=spvae2e1(iv2)
      acd76(22)=qshift(iv2)
      acd76(23)=spvak2e1(iv1)
      acd76(24)=spval4e1(iv1)
      acd76(25)=spvae1l4(iv1)
      acd76(26)=spvae1l5(iv1)
      acd76(27)=spvae1e2(iv1)
      acd76(28)=spvae2e1(iv1)
      acd76(29)=abb76(10)
      acd76(30)=abb76(12)
      acd76(31)=abb76(15)
      acd76(32)=spvae1k2(iv2)
      acd76(33)=abb76(29)
      acd76(34)=spvae1l3(iv2)
      acd76(35)=abb76(27)
      acd76(36)=spvae1k2(iv1)
      acd76(37)=spvae1l3(iv1)
      acd76(38)=abb76(34)
      acd76(39)=abb76(35)
      acd76(40)=spval3e1(iv2)
      acd76(41)=abb76(41)
      acd76(42)=spval3e1(iv1)
      acd76(43)=abb76(26)
      acd76(44)=abb76(38)
      acd76(45)=abb76(40)
      acd76(46)=abb76(19)
      acd76(47)=-acd76(13)*acd76(28)
      acd76(48)=acd76(11)*acd76(27)
      acd76(49)=acd76(7)*acd76(25)
      acd76(50)=acd76(5)*acd76(24)
      acd76(51)=-acd76(26)*acd76(9)
      acd76(52)=-acd76(23)*acd76(3)
      acd76(47)=acd76(52)+acd76(51)+acd76(50)+acd76(49)+acd76(47)+acd76(48)
      acd76(47)=acd76(22)*acd76(47)
      acd76(48)=-acd76(13)*acd76(21)
      acd76(49)=acd76(11)*acd76(20)
      acd76(50)=acd76(7)*acd76(18)
      acd76(51)=acd76(5)*acd76(17)
      acd76(52)=-acd76(19)*acd76(9)
      acd76(53)=-acd76(16)*acd76(3)
      acd76(48)=acd76(53)+acd76(52)+acd76(51)+acd76(50)+acd76(48)+acd76(49)
      acd76(48)=acd76(15)*acd76(48)
      acd76(49)=-acd76(13)*acd76(12)
      acd76(50)=acd76(11)*acd76(10)
      acd76(51)=-acd76(9)*acd76(8)
      acd76(52)=acd76(7)*acd76(6)
      acd76(53)=acd76(5)*acd76(4)
      acd76(54)=-acd76(3)*acd76(2)
      acd76(49)=acd76(54)+acd76(53)+acd76(52)+acd76(51)+acd76(50)+acd76(14)+acd&
      &76(49)
      acd76(49)=acd76(1)*acd76(49)
      acd76(47)=acd76(49)+acd76(47)+acd76(48)
      acd76(48)=acd76(33)*acd76(32)
      acd76(49)=acd76(34)*acd76(35)
      acd76(50)=acd76(20)*acd76(31)
      acd76(51)=acd76(18)*acd76(29)
      acd76(52)=acd76(19)*acd76(30)
      acd76(48)=acd76(52)+acd76(51)+acd76(50)+acd76(48)+acd76(49)
      acd76(48)=acd76(23)*acd76(48)
      acd76(49)=acd76(33)*acd76(36)
      acd76(50)=acd76(37)*acd76(35)
      acd76(51)=acd76(27)*acd76(31)
      acd76(52)=acd76(25)*acd76(29)
      acd76(53)=acd76(26)*acd76(30)
      acd76(49)=acd76(53)+acd76(52)+acd76(51)+acd76(49)+acd76(50)
      acd76(49)=acd76(16)*acd76(49)
      acd76(50)=-acd76(40)*acd76(44)
      acd76(51)=acd76(21)*acd76(43)
      acd76(52)=acd76(17)*acd76(38)
      acd76(50)=acd76(52)+acd76(50)+acd76(51)
      acd76(50)=acd76(26)*acd76(50)
      acd76(51)=-acd76(42)*acd76(44)
      acd76(52)=acd76(28)*acd76(43)
      acd76(53)=acd76(24)*acd76(38)
      acd76(51)=acd76(53)+acd76(51)+acd76(52)
      acd76(51)=acd76(19)*acd76(51)
      acd76(52)=acd76(20)*acd76(45)
      acd76(53)=acd76(18)*acd76(41)
      acd76(52)=acd76(53)+acd76(52)
      acd76(52)=acd76(42)*acd76(52)
      acd76(53)=acd76(27)*acd76(45)
      acd76(54)=acd76(25)*acd76(41)
      acd76(53)=acd76(54)+acd76(53)
      acd76(53)=acd76(40)*acd76(53)
      acd76(54)=-acd76(21)*acd76(46)
      acd76(55)=acd76(17)*acd76(39)
      acd76(54)=acd76(55)+acd76(54)
      acd76(54)=acd76(37)*acd76(54)
      acd76(55)=-acd76(28)*acd76(46)
      acd76(56)=acd76(24)*acd76(39)
      acd76(55)=acd76(56)+acd76(55)
      acd76(55)=acd76(34)*acd76(55)
      brack=2.0_ki*acd76(47)+acd76(48)+acd76(49)+acd76(50)+acd76(51)+acd76(52)+&
      &acd76(53)+acd76(54)+acd76(55)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd76h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd76
      complex(ki) :: brack
      acd76(1)=d(iv1,iv2)
      acd76(2)=spvak2e1(iv3)
      acd76(3)=abb76(28)
      acd76(4)=spval4e1(iv3)
      acd76(5)=abb76(20)
      acd76(6)=spvae1l4(iv3)
      acd76(7)=abb76(25)
      acd76(8)=spvae1l5(iv3)
      acd76(9)=abb76(24)
      acd76(10)=spvae1e2(iv3)
      acd76(11)=abb76(22)
      acd76(12)=spvae2e1(iv3)
      acd76(13)=abb76(13)
      acd76(14)=d(iv1,iv3)
      acd76(15)=spvak2e1(iv2)
      acd76(16)=spval4e1(iv2)
      acd76(17)=spvae1l4(iv2)
      acd76(18)=spvae1l5(iv2)
      acd76(19)=spvae1e2(iv2)
      acd76(20)=spvae2e1(iv2)
      acd76(21)=d(iv2,iv3)
      acd76(22)=spvak2e1(iv1)
      acd76(23)=spval4e1(iv1)
      acd76(24)=spvae1l4(iv1)
      acd76(25)=spvae1l5(iv1)
      acd76(26)=spvae1e2(iv1)
      acd76(27)=spvae2e1(iv1)
      acd76(28)=-acd76(2)*acd76(3)
      acd76(29)=acd76(4)*acd76(5)
      acd76(30)=acd76(6)*acd76(7)
      acd76(31)=-acd76(8)*acd76(9)
      acd76(32)=acd76(10)*acd76(11)
      acd76(33)=-acd76(12)*acd76(13)
      acd76(28)=acd76(33)+acd76(32)+acd76(31)+acd76(30)+acd76(28)+acd76(29)
      acd76(28)=acd76(1)*acd76(28)
      acd76(29)=-acd76(15)*acd76(3)
      acd76(30)=acd76(16)*acd76(5)
      acd76(31)=acd76(17)*acd76(7)
      acd76(32)=-acd76(18)*acd76(9)
      acd76(33)=acd76(19)*acd76(11)
      acd76(34)=-acd76(20)*acd76(13)
      acd76(29)=acd76(34)+acd76(33)+acd76(32)+acd76(31)+acd76(30)+acd76(29)
      acd76(29)=acd76(14)*acd76(29)
      acd76(30)=-acd76(22)*acd76(3)
      acd76(31)=acd76(23)*acd76(5)
      acd76(32)=acd76(24)*acd76(7)
      acd76(33)=-acd76(25)*acd76(9)
      acd76(34)=acd76(26)*acd76(11)
      acd76(35)=-acd76(27)*acd76(13)
      acd76(30)=acd76(35)+acd76(34)+acd76(33)+acd76(32)+acd76(31)+acd76(30)
      acd76(30)=acd76(21)*acd76(30)
      acd76(28)=acd76(30)+acd76(29)+acd76(28)
      brack=2.0_ki*acd76(28)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd76h8
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
      qshift = k3+k5
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
end module     p2_gg_httbar_d76h8l1d
