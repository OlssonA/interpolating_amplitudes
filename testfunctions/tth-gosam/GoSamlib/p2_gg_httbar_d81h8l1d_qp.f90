module     p2_gg_httbar_d81h8l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d81h8l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd81h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(69) :: acd81
      complex(ki) :: brack
      acd81(1)=dotproduct(k2,qshift)
      acd81(2)=abb81(19)
      acd81(3)=dotproduct(qshift,qshift)
      acd81(4)=abb81(22)
      acd81(5)=dotproduct(qshift,spvak2l3)
      acd81(6)=abb81(23)
      acd81(7)=dotproduct(qshift,spvak2l5)
      acd81(8)=abb81(21)
      acd81(9)=dotproduct(qshift,spval3k2)
      acd81(10)=abb81(56)
      acd81(11)=dotproduct(qshift,spval4k2)
      acd81(12)=abb81(18)
      acd81(13)=dotproduct(qshift,spvak1e2)
      acd81(14)=abb81(20)
      acd81(15)=dotproduct(qshift,spvae2k1)
      acd81(16)=abb81(16)
      acd81(17)=dotproduct(qshift,spvae1k2)
      acd81(18)=dotproduct(qshift,spvak2e2)
      acd81(19)=dotproduct(qshift,spvae2e1)
      acd81(20)=abb81(12)
      acd81(21)=abb81(51)
      acd81(22)=dotproduct(qshift,spvae1l3)
      acd81(23)=abb81(8)
      acd81(24)=abb81(30)
      acd81(25)=abb81(17)
      acd81(26)=dotproduct(qshift,spval3e2)
      acd81(27)=abb81(13)
      acd81(28)=dotproduct(qshift,spval4e2)
      acd81(29)=abb81(46)
      acd81(30)=abb81(15)
      acd81(31)=abb81(42)
      acd81(32)=dotproduct(qshift,spvae2k2)
      acd81(33)=dotproduct(qshift,spvae1e2)
      acd81(34)=abb81(11)
      acd81(35)=abb81(39)
      acd81(36)=dotproduct(qshift,spval3e1)
      acd81(37)=dotproduct(qshift,spvae2l5)
      acd81(38)=abb81(14)
      acd81(39)=abb81(38)
      acd81(40)=dotproduct(qshift,spval4e1)
      acd81(41)=abb81(10)
      acd81(42)=abb81(31)
      acd81(43)=dotproduct(qshift,spvae2l3)
      acd81(44)=abb81(43)
      acd81(45)=abb81(47)
      acd81(46)=abb81(24)
      acd81(47)=abb81(41)
      acd81(48)=abb81(44)
      acd81(49)=abb81(48)
      acd81(50)=abb81(45)
      acd81(51)=abb81(9)
      acd81(52)=acd81(38)*acd81(36)
      acd81(53)=acd81(41)*acd81(40)
      acd81(52)=acd81(42)+acd81(53)+acd81(52)
      acd81(52)=acd81(52)*acd81(37)
      acd81(53)=acd81(34)*acd81(32)
      acd81(54)=acd81(39)*acd81(36)
      acd81(55)=acd81(44)*acd81(43)
      acd81(56)=acd81(45)*acd81(40)
      acd81(52)=acd81(56)+acd81(54)+acd81(52)-acd81(46)+acd81(55)+acd81(53)
      acd81(52)=acd81(33)*acd81(52)
      acd81(53)=-acd81(20)*acd81(17)
      acd81(54)=-acd81(23)*acd81(22)
      acd81(53)=acd81(24)+acd81(54)+acd81(53)
      acd81(53)=acd81(53)*acd81(19)
      acd81(53)=-acd81(25)+acd81(53)
      acd81(53)=acd81(18)*acd81(53)
      acd81(54)=acd81(27)*acd81(26)
      acd81(55)=acd81(29)*acd81(28)
      acd81(54)=-acd81(30)+acd81(55)+acd81(54)
      acd81(54)=acd81(19)*acd81(54)
      acd81(55)=-acd81(2)*acd81(1)
      acd81(56)=acd81(4)*acd81(3)
      acd81(57)=-acd81(6)*acd81(5)
      acd81(58)=-acd81(8)*acd81(7)
      acd81(59)=-acd81(10)*acd81(9)
      acd81(60)=-acd81(12)*acd81(11)
      acd81(61)=-acd81(14)*acd81(13)
      acd81(62)=-acd81(16)*acd81(15)
      acd81(63)=acd81(21)*acd81(17)
      acd81(64)=-acd81(31)*acd81(22)
      acd81(65)=-acd81(35)*acd81(32)
      acd81(66)=-acd81(47)*acd81(36)
      acd81(67)=-acd81(48)*acd81(37)
      acd81(68)=-acd81(49)*acd81(43)
      acd81(69)=-acd81(50)*acd81(40)
      brack=acd81(51)+acd81(52)+acd81(53)+acd81(54)+acd81(55)+acd81(56)+acd81(5&
      &7)+acd81(58)+acd81(59)+acd81(60)+acd81(61)+acd81(62)+acd81(63)+acd81(64)&
      &+acd81(65)+acd81(66)+acd81(67)+acd81(68)+acd81(69)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd81h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(82) :: acd81
      complex(ki) :: brack
      acd81(1)=k2(iv1)
      acd81(2)=abb81(19)
      acd81(3)=qshift(iv1)
      acd81(4)=abb81(22)
      acd81(5)=spvak2l3(iv1)
      acd81(6)=abb81(23)
      acd81(7)=spvak2l5(iv1)
      acd81(8)=abb81(21)
      acd81(9)=spval3k2(iv1)
      acd81(10)=abb81(56)
      acd81(11)=spval4k2(iv1)
      acd81(12)=abb81(18)
      acd81(13)=spvak1e2(iv1)
      acd81(14)=abb81(20)
      acd81(15)=spvae2k1(iv1)
      acd81(16)=abb81(16)
      acd81(17)=spvae1k2(iv1)
      acd81(18)=dotproduct(qshift,spvak2e2)
      acd81(19)=dotproduct(qshift,spvae2e1)
      acd81(20)=abb81(12)
      acd81(21)=abb81(51)
      acd81(22)=spvak2e2(iv1)
      acd81(23)=dotproduct(qshift,spvae1k2)
      acd81(24)=dotproduct(qshift,spvae1l3)
      acd81(25)=abb81(8)
      acd81(26)=abb81(30)
      acd81(27)=abb81(17)
      acd81(28)=spvae2e1(iv1)
      acd81(29)=dotproduct(qshift,spval3e2)
      acd81(30)=abb81(13)
      acd81(31)=dotproduct(qshift,spval4e2)
      acd81(32)=abb81(46)
      acd81(33)=abb81(15)
      acd81(34)=spvae1l3(iv1)
      acd81(35)=abb81(42)
      acd81(36)=spvae2k2(iv1)
      acd81(37)=dotproduct(qshift,spvae1e2)
      acd81(38)=abb81(11)
      acd81(39)=abb81(39)
      acd81(40)=spvae1e2(iv1)
      acd81(41)=dotproduct(qshift,spvae2k2)
      acd81(42)=dotproduct(qshift,spval3e1)
      acd81(43)=dotproduct(qshift,spvae2l5)
      acd81(44)=abb81(14)
      acd81(45)=abb81(38)
      acd81(46)=dotproduct(qshift,spval4e1)
      acd81(47)=abb81(10)
      acd81(48)=abb81(31)
      acd81(49)=dotproduct(qshift,spvae2l3)
      acd81(50)=abb81(43)
      acd81(51)=abb81(47)
      acd81(52)=abb81(24)
      acd81(53)=spval3e1(iv1)
      acd81(54)=abb81(41)
      acd81(55)=spvae2l5(iv1)
      acd81(56)=abb81(44)
      acd81(57)=spval3e2(iv1)
      acd81(58)=spvae2l3(iv1)
      acd81(59)=abb81(48)
      acd81(60)=spval4e1(iv1)
      acd81(61)=abb81(45)
      acd81(62)=spval4e2(iv1)
      acd81(63)=acd81(46)*acd81(47)
      acd81(64)=acd81(42)*acd81(44)
      acd81(63)=acd81(48)+acd81(63)+acd81(64)
      acd81(64)=acd81(55)*acd81(63)
      acd81(65)=acd81(47)*acd81(60)
      acd81(66)=acd81(44)*acd81(53)
      acd81(65)=acd81(65)+acd81(66)
      acd81(65)=acd81(43)*acd81(65)
      acd81(66)=acd81(50)*acd81(58)
      acd81(67)=acd81(36)*acd81(38)
      acd81(68)=acd81(60)*acd81(51)
      acd81(69)=acd81(53)*acd81(45)
      acd81(64)=acd81(65)+acd81(64)+acd81(69)+acd81(68)+acd81(66)+acd81(67)
      acd81(64)=acd81(37)*acd81(64)
      acd81(63)=acd81(43)*acd81(63)
      acd81(65)=acd81(50)*acd81(49)
      acd81(66)=acd81(38)*acd81(41)
      acd81(67)=acd81(46)*acd81(51)
      acd81(68)=acd81(42)*acd81(45)
      acd81(63)=acd81(63)+acd81(68)+acd81(67)+acd81(66)-acd81(52)+acd81(65)
      acd81(63)=acd81(40)*acd81(63)
      acd81(65)=acd81(25)*acd81(24)
      acd81(66)=acd81(20)*acd81(23)
      acd81(65)=-acd81(26)+acd81(65)+acd81(66)
      acd81(66)=-acd81(22)*acd81(65)
      acd81(67)=-acd81(25)*acd81(34)
      acd81(68)=-acd81(20)*acd81(17)
      acd81(67)=acd81(67)+acd81(68)
      acd81(67)=acd81(18)*acd81(67)
      acd81(68)=acd81(32)*acd81(62)
      acd81(69)=acd81(30)*acd81(57)
      acd81(66)=acd81(67)+acd81(66)+acd81(68)+acd81(69)
      acd81(66)=acd81(19)*acd81(66)
      acd81(65)=-acd81(18)*acd81(65)
      acd81(67)=acd81(32)*acd81(31)
      acd81(68)=acd81(30)*acd81(29)
      acd81(65)=acd81(65)+acd81(68)-acd81(33)+acd81(67)
      acd81(65)=acd81(28)*acd81(65)
      acd81(67)=-acd81(15)*acd81(16)
      acd81(68)=-acd81(13)*acd81(14)
      acd81(69)=-acd81(11)*acd81(12)
      acd81(70)=-acd81(9)*acd81(10)
      acd81(71)=-acd81(7)*acd81(8)
      acd81(72)=-acd81(5)*acd81(6)
      acd81(73)=acd81(3)*acd81(4)
      acd81(74)=-acd81(1)*acd81(2)
      acd81(75)=-acd81(58)*acd81(59)
      acd81(76)=-acd81(36)*acd81(39)
      acd81(77)=-acd81(34)*acd81(35)
      acd81(78)=acd81(17)*acd81(21)
      acd81(79)=-acd81(60)*acd81(61)
      acd81(80)=-acd81(53)*acd81(54)
      acd81(81)=-acd81(55)*acd81(56)
      acd81(82)=-acd81(22)*acd81(27)
      brack=acd81(63)+acd81(64)+acd81(65)+acd81(66)+acd81(67)+acd81(68)+acd81(6&
      &9)+acd81(70)+acd81(71)+acd81(72)+2.0_ki*acd81(73)+acd81(74)+acd81(75)+ac&
      &d81(76)+acd81(77)+acd81(78)+acd81(79)+acd81(80)+acd81(81)+acd81(82)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd81h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(61) :: acd81
      complex(ki) :: brack
      acd81(1)=d(iv1,iv2)
      acd81(2)=abb81(22)
      acd81(3)=spvae1k2(iv1)
      acd81(4)=spvak2e2(iv2)
      acd81(5)=dotproduct(qshift,spvae2e1)
      acd81(6)=abb81(12)
      acd81(7)=spvae2e1(iv2)
      acd81(8)=dotproduct(qshift,spvak2e2)
      acd81(9)=spvae1k2(iv2)
      acd81(10)=spvak2e2(iv1)
      acd81(11)=spvae2e1(iv1)
      acd81(12)=dotproduct(qshift,spvae1k2)
      acd81(13)=dotproduct(qshift,spvae1l3)
      acd81(14)=abb81(8)
      acd81(15)=abb81(30)
      acd81(16)=spvae1l3(iv2)
      acd81(17)=spvae1l3(iv1)
      acd81(18)=spval3e2(iv2)
      acd81(19)=abb81(13)
      acd81(20)=spval4e2(iv2)
      acd81(21)=abb81(46)
      acd81(22)=spval3e2(iv1)
      acd81(23)=spval4e2(iv1)
      acd81(24)=spvae2k2(iv1)
      acd81(25)=spvae1e2(iv2)
      acd81(26)=abb81(11)
      acd81(27)=spvae2k2(iv2)
      acd81(28)=spvae1e2(iv1)
      acd81(29)=spval3e1(iv2)
      acd81(30)=dotproduct(qshift,spvae2l5)
      acd81(31)=abb81(14)
      acd81(32)=abb81(38)
      acd81(33)=spvae2l5(iv2)
      acd81(34)=dotproduct(qshift,spval3e1)
      acd81(35)=dotproduct(qshift,spval4e1)
      acd81(36)=abb81(10)
      acd81(37)=abb81(31)
      acd81(38)=spvae2l3(iv2)
      acd81(39)=abb81(43)
      acd81(40)=spval4e1(iv2)
      acd81(41)=abb81(47)
      acd81(42)=spval3e1(iv1)
      acd81(43)=spvae2l5(iv1)
      acd81(44)=spvae2l3(iv1)
      acd81(45)=spval4e1(iv1)
      acd81(46)=dotproduct(qshift,spvae1e2)
      acd81(47)=-acd81(3)*acd81(6)
      acd81(48)=-acd81(17)*acd81(14)
      acd81(47)=acd81(48)+acd81(47)
      acd81(48)=acd81(5)*acd81(4)
      acd81(49)=acd81(8)*acd81(7)
      acd81(48)=acd81(48)+acd81(49)
      acd81(47)=acd81(48)*acd81(47)
      acd81(48)=-acd81(9)*acd81(6)
      acd81(49)=-acd81(16)*acd81(14)
      acd81(48)=acd81(49)+acd81(48)
      acd81(49)=acd81(5)*acd81(10)
      acd81(50)=acd81(8)*acd81(11)
      acd81(49)=acd81(49)+acd81(50)
      acd81(48)=acd81(49)*acd81(48)
      acd81(49)=-acd81(12)*acd81(6)
      acd81(50)=-acd81(13)*acd81(14)
      acd81(49)=acd81(15)+acd81(50)+acd81(49)
      acd81(50)=acd81(4)*acd81(11)
      acd81(51)=acd81(10)*acd81(7)
      acd81(50)=acd81(50)+acd81(51)
      acd81(49)=acd81(50)*acd81(49)
      acd81(50)=acd81(34)*acd81(31)
      acd81(51)=acd81(35)*acd81(36)
      acd81(50)=acd81(37)+acd81(51)+acd81(50)
      acd81(51)=acd81(33)*acd81(28)
      acd81(52)=acd81(43)*acd81(25)
      acd81(51)=acd81(51)+acd81(52)
      acd81(50)=acd81(51)*acd81(50)
      acd81(51)=acd81(18)*acd81(11)
      acd81(52)=acd81(22)*acd81(7)
      acd81(51)=acd81(52)+acd81(51)
      acd81(51)=acd81(19)*acd81(51)
      acd81(52)=acd81(20)*acd81(11)
      acd81(53)=acd81(23)*acd81(7)
      acd81(52)=acd81(53)+acd81(52)
      acd81(52)=acd81(21)*acd81(52)
      acd81(53)=acd81(24)*acd81(25)
      acd81(54)=acd81(27)*acd81(28)
      acd81(53)=acd81(54)+acd81(53)
      acd81(53)=acd81(26)*acd81(53)
      acd81(54)=acd81(38)*acd81(28)
      acd81(55)=acd81(44)*acd81(25)
      acd81(54)=acd81(55)+acd81(54)
      acd81(54)=acd81(39)*acd81(54)
      acd81(55)=acd81(30)*acd81(31)
      acd81(56)=acd81(28)*acd81(55)
      acd81(57)=acd81(46)*acd81(43)
      acd81(58)=acd81(31)*acd81(57)
      acd81(56)=acd81(56)+acd81(58)
      acd81(56)=acd81(29)*acd81(56)
      acd81(58)=acd81(30)*acd81(36)
      acd81(59)=acd81(28)*acd81(58)
      acd81(57)=acd81(36)*acd81(57)
      acd81(57)=acd81(59)+acd81(57)
      acd81(57)=acd81(40)*acd81(57)
      acd81(55)=acd81(25)*acd81(55)
      acd81(59)=acd81(46)*acd81(33)
      acd81(60)=acd81(31)*acd81(59)
      acd81(55)=acd81(55)+acd81(60)
      acd81(55)=acd81(42)*acd81(55)
      acd81(58)=acd81(25)*acd81(58)
      acd81(59)=acd81(36)*acd81(59)
      acd81(58)=acd81(58)+acd81(59)
      acd81(58)=acd81(45)*acd81(58)
      acd81(59)=acd81(29)*acd81(28)
      acd81(60)=acd81(42)*acd81(25)
      acd81(59)=acd81(59)+acd81(60)
      acd81(59)=acd81(32)*acd81(59)
      acd81(60)=acd81(40)*acd81(28)
      acd81(61)=acd81(45)*acd81(25)
      acd81(60)=acd81(60)+acd81(61)
      acd81(60)=acd81(41)*acd81(60)
      acd81(61)=acd81(2)*acd81(1)
      brack=acd81(47)+acd81(48)+acd81(49)+acd81(50)+acd81(51)+acd81(52)+acd81(5&
      &3)+acd81(54)+acd81(55)+acd81(56)+acd81(57)+acd81(58)+acd81(59)+acd81(60)&
      &+2.0_ki*acd81(61)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd81h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd81
      complex(ki) :: brack
      acd81(1)=spvae1k2(iv1)
      acd81(2)=spvak2e2(iv2)
      acd81(3)=spvae2e1(iv3)
      acd81(4)=abb81(12)
      acd81(5)=spvak2e2(iv3)
      acd81(6)=spvae2e1(iv2)
      acd81(7)=spvae1k2(iv2)
      acd81(8)=spvak2e2(iv1)
      acd81(9)=spvae2e1(iv1)
      acd81(10)=spvae1k2(iv3)
      acd81(11)=spvae1l3(iv3)
      acd81(12)=abb81(8)
      acd81(13)=spvae1l3(iv2)
      acd81(14)=spvae1l3(iv1)
      acd81(15)=spval3e1(iv1)
      acd81(16)=spvae2l5(iv2)
      acd81(17)=spvae1e2(iv3)
      acd81(18)=abb81(14)
      acd81(19)=spvae2l5(iv3)
      acd81(20)=spvae1e2(iv2)
      acd81(21)=spval3e1(iv2)
      acd81(22)=spvae2l5(iv1)
      acd81(23)=spvae1e2(iv1)
      acd81(24)=spval3e1(iv3)
      acd81(25)=spval4e1(iv3)
      acd81(26)=abb81(10)
      acd81(27)=spval4e1(iv2)
      acd81(28)=spval4e1(iv1)
      acd81(29)=acd81(3)*acd81(2)
      acd81(30)=acd81(6)*acd81(5)
      acd81(29)=acd81(29)+acd81(30)
      acd81(30)=-acd81(1)*acd81(29)
      acd81(31)=acd81(8)*acd81(3)
      acd81(32)=acd81(9)*acd81(5)
      acd81(31)=acd81(31)+acd81(32)
      acd81(32)=-acd81(7)*acd81(31)
      acd81(33)=acd81(8)*acd81(6)
      acd81(34)=acd81(9)*acd81(2)
      acd81(33)=acd81(33)+acd81(34)
      acd81(34)=-acd81(10)*acd81(33)
      acd81(30)=acd81(34)+acd81(30)+acd81(32)
      acd81(30)=acd81(4)*acd81(30)
      acd81(32)=-acd81(11)*acd81(33)
      acd81(31)=-acd81(13)*acd81(31)
      acd81(29)=-acd81(14)*acd81(29)
      acd81(29)=acd81(29)+acd81(31)+acd81(32)
      acd81(29)=acd81(12)*acd81(29)
      acd81(31)=acd81(17)*acd81(16)
      acd81(32)=acd81(20)*acd81(19)
      acd81(31)=acd81(31)+acd81(32)
      acd81(32)=acd81(15)*acd81(31)
      acd81(33)=acd81(22)*acd81(17)
      acd81(34)=acd81(23)*acd81(19)
      acd81(33)=acd81(33)+acd81(34)
      acd81(34)=acd81(21)*acd81(33)
      acd81(35)=acd81(22)*acd81(20)
      acd81(36)=acd81(23)*acd81(16)
      acd81(35)=acd81(35)+acd81(36)
      acd81(36)=acd81(24)*acd81(35)
      acd81(32)=acd81(36)+acd81(34)+acd81(32)
      acd81(32)=acd81(18)*acd81(32)
      acd81(34)=acd81(25)*acd81(35)
      acd81(33)=acd81(27)*acd81(33)
      acd81(31)=acd81(28)*acd81(31)
      acd81(31)=acd81(31)+acd81(33)+acd81(34)
      acd81(31)=acd81(26)*acd81(31)
      brack=acd81(29)+acd81(30)+acd81(31)+acd81(32)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd81h8_qp
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
      qshift = k2-k3-k4
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
end module     p2_gg_httbar_d81h8l1d_qp
