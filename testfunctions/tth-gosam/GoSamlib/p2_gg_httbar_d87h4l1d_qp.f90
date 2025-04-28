module     p2_gg_httbar_d87h4l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d87h4l1d_qp.f90
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
      use p2_gg_httbar_abbrevd87h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(64) :: acd87
      complex(ki) :: brack
      acd87(1)=dotproduct(qshift,qshift)
      acd87(2)=abb87(16)
      acd87(3)=dotproduct(qshift,spvak1k2)
      acd87(4)=abb87(23)
      acd87(5)=dotproduct(qshift,spvak1l3)
      acd87(6)=abb87(46)
      acd87(7)=dotproduct(qshift,spvak1l4)
      acd87(8)=abb87(17)
      acd87(9)=dotproduct(qshift,spvak2k1)
      acd87(10)=abb87(10)
      acd87(11)=dotproduct(qshift,spval3k1)
      acd87(12)=abb87(30)
      acd87(13)=dotproduct(qshift,spval5k1)
      acd87(14)=abb87(12)
      acd87(15)=dotproduct(qshift,spvak2e1)
      acd87(16)=dotproduct(qshift,spvae1e2)
      acd87(17)=abb87(9)
      acd87(18)=abb87(25)
      acd87(19)=dotproduct(qshift,spval3e1)
      acd87(20)=dotproduct(qshift,spvae2l4)
      acd87(21)=abb87(15)
      acd87(22)=abb87(19)
      acd87(23)=dotproduct(qshift,spval5e1)
      acd87(24)=abb87(22)
      acd87(25)=abb87(26)
      acd87(26)=abb87(33)
      acd87(27)=abb87(31)
      acd87(28)=dotproduct(qshift,spvae1k2)
      acd87(29)=dotproduct(qshift,spvak2e2)
      acd87(30)=dotproduct(qshift,spvae2e1)
      acd87(31)=abb87(11)
      acd87(32)=abb87(20)
      acd87(33)=abb87(18)
      acd87(34)=dotproduct(qshift,spvae1l3)
      acd87(35)=abb87(51)
      acd87(36)=abb87(45)
      acd87(37)=abb87(28)
      acd87(38)=abb87(27)
      acd87(39)=dotproduct(qshift,spvae1l4)
      acd87(40)=abb87(35)
      acd87(41)=abb87(24)
      acd87(42)=abb87(13)
      acd87(43)=abb87(38)
      acd87(44)=abb87(21)
      acd87(45)=abb87(39)
      acd87(46)=abb87(14)
      acd87(47)=abb87(8)
      acd87(48)=-acd87(21)*acd87(19)
      acd87(49)=-acd87(24)*acd87(23)
      acd87(48)=acd87(25)+acd87(49)+acd87(48)
      acd87(48)=acd87(48)*acd87(20)
      acd87(49)=acd87(22)*acd87(19)
      acd87(50)=acd87(26)*acd87(23)
      acd87(48)=-acd87(27)+acd87(50)+acd87(49)+acd87(48)
      acd87(48)=acd87(16)*acd87(48)
      acd87(49)=-acd87(31)*acd87(28)
      acd87(50)=-acd87(35)*acd87(34)
      acd87(49)=acd87(36)+acd87(50)+acd87(49)
      acd87(49)=acd87(49)*acd87(29)
      acd87(50)=acd87(32)*acd87(28)
      acd87(51)=acd87(38)*acd87(34)
      acd87(49)=-acd87(41)+acd87(51)+acd87(50)+acd87(49)
      acd87(49)=acd87(30)*acd87(49)
      acd87(50)=acd87(17)*acd87(16)
      acd87(50)=-acd87(18)+acd87(50)
      acd87(50)=acd87(15)*acd87(50)
      acd87(51)=acd87(40)*acd87(30)
      acd87(51)=-acd87(45)+acd87(51)
      acd87(51)=acd87(39)*acd87(51)
      acd87(52)=acd87(2)*acd87(1)
      acd87(53)=-acd87(4)*acd87(3)
      acd87(54)=-acd87(6)*acd87(5)
      acd87(55)=-acd87(8)*acd87(7)
      acd87(56)=-acd87(10)*acd87(9)
      acd87(57)=acd87(12)*acd87(11)
      acd87(58)=-acd87(14)*acd87(13)
      acd87(59)=-acd87(33)*acd87(28)
      acd87(60)=-acd87(37)*acd87(29)
      acd87(61)=-acd87(42)*acd87(34)
      acd87(62)=-acd87(43)*acd87(19)
      acd87(63)=-acd87(44)*acd87(20)
      acd87(64)=-acd87(46)*acd87(23)
      brack=acd87(47)+acd87(48)+acd87(49)+acd87(50)+acd87(51)+acd87(52)+acd87(5&
      &3)+acd87(54)+acd87(55)+acd87(56)+acd87(57)+acd87(58)+acd87(59)+acd87(60)&
      &+acd87(61)+acd87(62)+acd87(63)+acd87(64)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd87h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(75) :: acd87
      complex(ki) :: brack
      acd87(1)=qshift(iv1)
      acd87(2)=abb87(16)
      acd87(3)=spvak1k2(iv1)
      acd87(4)=abb87(23)
      acd87(5)=spvak1l3(iv1)
      acd87(6)=abb87(46)
      acd87(7)=spvak1l4(iv1)
      acd87(8)=abb87(17)
      acd87(9)=spvak2k1(iv1)
      acd87(10)=abb87(10)
      acd87(11)=spval3k1(iv1)
      acd87(12)=abb87(30)
      acd87(13)=spval5k1(iv1)
      acd87(14)=abb87(12)
      acd87(15)=spvak2e1(iv1)
      acd87(16)=dotproduct(qshift,spvae1e2)
      acd87(17)=abb87(9)
      acd87(18)=abb87(25)
      acd87(19)=spvae1e2(iv1)
      acd87(20)=dotproduct(qshift,spvak2e1)
      acd87(21)=dotproduct(qshift,spval3e1)
      acd87(22)=dotproduct(qshift,spvae2l4)
      acd87(23)=abb87(15)
      acd87(24)=abb87(19)
      acd87(25)=dotproduct(qshift,spval5e1)
      acd87(26)=abb87(22)
      acd87(27)=abb87(26)
      acd87(28)=abb87(33)
      acd87(29)=abb87(31)
      acd87(30)=spvae1k2(iv1)
      acd87(31)=dotproduct(qshift,spvak2e2)
      acd87(32)=dotproduct(qshift,spvae2e1)
      acd87(33)=abb87(11)
      acd87(34)=abb87(20)
      acd87(35)=abb87(18)
      acd87(36)=spvak2e2(iv1)
      acd87(37)=dotproduct(qshift,spvae1k2)
      acd87(38)=dotproduct(qshift,spvae1l3)
      acd87(39)=abb87(51)
      acd87(40)=abb87(45)
      acd87(41)=abb87(28)
      acd87(42)=spvae2e1(iv1)
      acd87(43)=abb87(27)
      acd87(44)=dotproduct(qshift,spvae1l4)
      acd87(45)=abb87(35)
      acd87(46)=abb87(24)
      acd87(47)=spvae1l3(iv1)
      acd87(48)=abb87(13)
      acd87(49)=spval3e1(iv1)
      acd87(50)=abb87(38)
      acd87(51)=spvae2l4(iv1)
      acd87(52)=abb87(21)
      acd87(53)=spvae1l4(iv1)
      acd87(54)=abb87(39)
      acd87(55)=spval5e1(iv1)
      acd87(56)=abb87(14)
      acd87(57)=acd87(38)*acd87(39)
      acd87(58)=acd87(33)*acd87(37)
      acd87(57)=-acd87(40)+acd87(57)+acd87(58)
      acd87(58)=acd87(36)*acd87(57)
      acd87(59)=acd87(39)*acd87(47)
      acd87(60)=acd87(30)*acd87(33)
      acd87(59)=acd87(59)+acd87(60)
      acd87(59)=acd87(31)*acd87(59)
      acd87(60)=-acd87(45)*acd87(53)
      acd87(61)=-acd87(47)*acd87(43)
      acd87(62)=-acd87(30)*acd87(34)
      acd87(58)=acd87(59)+acd87(58)+acd87(62)+acd87(60)+acd87(61)
      acd87(58)=acd87(32)*acd87(58)
      acd87(59)=acd87(25)*acd87(26)
      acd87(60)=acd87(21)*acd87(23)
      acd87(59)=-acd87(27)+acd87(59)+acd87(60)
      acd87(60)=acd87(51)*acd87(59)
      acd87(61)=acd87(26)*acd87(55)
      acd87(62)=acd87(23)*acd87(49)
      acd87(61)=acd87(61)+acd87(62)
      acd87(61)=acd87(22)*acd87(61)
      acd87(62)=-acd87(15)*acd87(17)
      acd87(63)=-acd87(55)*acd87(28)
      acd87(64)=-acd87(49)*acd87(24)
      acd87(60)=acd87(61)+acd87(60)+acd87(64)+acd87(62)+acd87(63)
      acd87(60)=acd87(16)*acd87(60)
      acd87(57)=acd87(31)*acd87(57)
      acd87(61)=-acd87(45)*acd87(44)
      acd87(62)=-acd87(38)*acd87(43)
      acd87(63)=-acd87(37)*acd87(34)
      acd87(57)=acd87(57)+acd87(63)+acd87(62)+acd87(46)+acd87(61)
      acd87(57)=acd87(42)*acd87(57)
      acd87(59)=acd87(22)*acd87(59)
      acd87(61)=-acd87(17)*acd87(20)
      acd87(62)=-acd87(25)*acd87(28)
      acd87(63)=-acd87(21)*acd87(24)
      acd87(59)=acd87(59)+acd87(63)+acd87(62)+acd87(29)+acd87(61)
      acd87(59)=acd87(19)*acd87(59)
      acd87(61)=acd87(13)*acd87(14)
      acd87(62)=-acd87(11)*acd87(12)
      acd87(63)=acd87(9)*acd87(10)
      acd87(64)=acd87(7)*acd87(8)
      acd87(65)=acd87(5)*acd87(6)
      acd87(66)=acd87(3)*acd87(4)
      acd87(67)=acd87(1)*acd87(2)
      acd87(68)=acd87(53)*acd87(54)
      acd87(69)=acd87(15)*acd87(18)
      acd87(70)=acd87(55)*acd87(56)
      acd87(71)=acd87(49)*acd87(50)
      acd87(72)=acd87(47)*acd87(48)
      acd87(73)=acd87(30)*acd87(35)
      acd87(74)=acd87(51)*acd87(52)
      acd87(75)=acd87(36)*acd87(41)
      brack=acd87(57)+acd87(58)+acd87(59)+acd87(60)+acd87(61)+acd87(62)+acd87(6&
      &3)+acd87(64)+acd87(65)+acd87(66)-2.0_ki*acd87(67)+acd87(68)+acd87(69)+ac&
      &d87(70)+acd87(71)+acd87(72)+acd87(73)+acd87(74)+acd87(75)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd87h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(57) :: acd87
      complex(ki) :: brack
      acd87(1)=d(iv1,iv2)
      acd87(2)=abb87(16)
      acd87(3)=spvak2e1(iv1)
      acd87(4)=spvae1e2(iv2)
      acd87(5)=abb87(9)
      acd87(6)=spvak2e1(iv2)
      acd87(7)=spvae1e2(iv1)
      acd87(8)=spval3e1(iv2)
      acd87(9)=dotproduct(qshift,spvae2l4)
      acd87(10)=abb87(15)
      acd87(11)=abb87(19)
      acd87(12)=spvae2l4(iv2)
      acd87(13)=dotproduct(qshift,spval3e1)
      acd87(14)=dotproduct(qshift,spval5e1)
      acd87(15)=abb87(22)
      acd87(16)=abb87(26)
      acd87(17)=spval5e1(iv2)
      acd87(18)=abb87(33)
      acd87(19)=spval3e1(iv1)
      acd87(20)=spvae2l4(iv1)
      acd87(21)=spval5e1(iv1)
      acd87(22)=spvae1k2(iv1)
      acd87(23)=spvak2e2(iv2)
      acd87(24)=dotproduct(qshift,spvae2e1)
      acd87(25)=abb87(11)
      acd87(26)=spvae2e1(iv2)
      acd87(27)=dotproduct(qshift,spvak2e2)
      acd87(28)=abb87(20)
      acd87(29)=spvae1k2(iv2)
      acd87(30)=spvak2e2(iv1)
      acd87(31)=spvae2e1(iv1)
      acd87(32)=dotproduct(qshift,spvae1k2)
      acd87(33)=dotproduct(qshift,spvae1l3)
      acd87(34)=abb87(51)
      acd87(35)=abb87(45)
      acd87(36)=spvae1l3(iv2)
      acd87(37)=spvae1l3(iv1)
      acd87(38)=abb87(27)
      acd87(39)=spvae1l4(iv2)
      acd87(40)=abb87(35)
      acd87(41)=spvae1l4(iv1)
      acd87(42)=dotproduct(qshift,spvae1e2)
      acd87(43)=-acd87(19)*acd87(10)
      acd87(44)=-acd87(21)*acd87(15)
      acd87(43)=acd87(44)+acd87(43)
      acd87(44)=acd87(9)*acd87(4)
      acd87(45)=acd87(42)*acd87(12)
      acd87(44)=acd87(44)+acd87(45)
      acd87(43)=acd87(44)*acd87(43)
      acd87(44)=-acd87(8)*acd87(10)
      acd87(45)=-acd87(17)*acd87(15)
      acd87(44)=acd87(44)+acd87(45)
      acd87(45)=acd87(9)*acd87(7)
      acd87(46)=acd87(42)*acd87(20)
      acd87(45)=acd87(45)+acd87(46)
      acd87(44)=acd87(45)*acd87(44)
      acd87(45)=-acd87(13)*acd87(10)
      acd87(46)=-acd87(14)*acd87(15)
      acd87(45)=acd87(16)+acd87(46)+acd87(45)
      acd87(46)=acd87(12)*acd87(7)
      acd87(47)=acd87(20)*acd87(4)
      acd87(46)=acd87(46)+acd87(47)
      acd87(45)=acd87(46)*acd87(45)
      acd87(46)=-acd87(32)*acd87(25)
      acd87(47)=-acd87(33)*acd87(34)
      acd87(46)=acd87(35)+acd87(47)+acd87(46)
      acd87(47)=acd87(23)*acd87(31)
      acd87(48)=acd87(30)*acd87(26)
      acd87(47)=acd87(47)+acd87(48)
      acd87(46)=acd87(47)*acd87(46)
      acd87(47)=acd87(3)*acd87(4)
      acd87(48)=acd87(6)*acd87(7)
      acd87(47)=acd87(48)+acd87(47)
      acd87(47)=acd87(5)*acd87(47)
      acd87(48)=acd87(39)*acd87(31)
      acd87(49)=acd87(41)*acd87(26)
      acd87(48)=acd87(49)+acd87(48)
      acd87(48)=acd87(40)*acd87(48)
      acd87(49)=acd87(24)*acd87(23)
      acd87(50)=-acd87(25)*acd87(49)
      acd87(51)=acd87(27)*acd87(25)
      acd87(52)=-acd87(26)*acd87(51)
      acd87(50)=acd87(50)+acd87(52)
      acd87(50)=acd87(22)*acd87(50)
      acd87(52)=acd87(24)*acd87(30)
      acd87(53)=-acd87(25)*acd87(52)
      acd87(51)=-acd87(31)*acd87(51)
      acd87(51)=acd87(53)+acd87(51)
      acd87(51)=acd87(29)*acd87(51)
      acd87(52)=-acd87(34)*acd87(52)
      acd87(53)=acd87(27)*acd87(34)
      acd87(54)=-acd87(31)*acd87(53)
      acd87(52)=acd87(52)+acd87(54)
      acd87(52)=acd87(36)*acd87(52)
      acd87(49)=-acd87(34)*acd87(49)
      acd87(53)=-acd87(26)*acd87(53)
      acd87(49)=acd87(49)+acd87(53)
      acd87(49)=acd87(37)*acd87(49)
      acd87(53)=acd87(8)*acd87(7)
      acd87(54)=acd87(19)*acd87(4)
      acd87(53)=acd87(53)+acd87(54)
      acd87(53)=acd87(11)*acd87(53)
      acd87(54)=acd87(17)*acd87(7)
      acd87(55)=acd87(21)*acd87(4)
      acd87(54)=acd87(54)+acd87(55)
      acd87(54)=acd87(18)*acd87(54)
      acd87(55)=acd87(22)*acd87(26)
      acd87(56)=acd87(29)*acd87(31)
      acd87(55)=acd87(55)+acd87(56)
      acd87(55)=acd87(28)*acd87(55)
      acd87(56)=acd87(36)*acd87(31)
      acd87(57)=acd87(37)*acd87(26)
      acd87(56)=acd87(56)+acd87(57)
      acd87(56)=acd87(38)*acd87(56)
      acd87(57)=acd87(2)*acd87(1)
      brack=acd87(43)+acd87(44)+acd87(45)+acd87(46)+acd87(47)+acd87(48)+acd87(4&
      &9)+acd87(50)+acd87(51)+acd87(52)+acd87(53)+acd87(54)+acd87(55)+acd87(56)&
      &+2.0_ki*acd87(57)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd87h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd87
      complex(ki) :: brack
      acd87(1)=spvae1k2(iv1)
      acd87(2)=spvak2e2(iv2)
      acd87(3)=spvae2e1(iv3)
      acd87(4)=abb87(11)
      acd87(5)=spvak2e2(iv3)
      acd87(6)=spvae2e1(iv2)
      acd87(7)=spvae1k2(iv2)
      acd87(8)=spvak2e2(iv1)
      acd87(9)=spvae2e1(iv1)
      acd87(10)=spvae1k2(iv3)
      acd87(11)=spvae1l3(iv3)
      acd87(12)=abb87(51)
      acd87(13)=spvae1l3(iv2)
      acd87(14)=spvae1l3(iv1)
      acd87(15)=spval3e1(iv1)
      acd87(16)=spvae2l4(iv2)
      acd87(17)=spvae1e2(iv3)
      acd87(18)=abb87(15)
      acd87(19)=spvae2l4(iv3)
      acd87(20)=spvae1e2(iv2)
      acd87(21)=spval3e1(iv2)
      acd87(22)=spvae2l4(iv1)
      acd87(23)=spvae1e2(iv1)
      acd87(24)=spval3e1(iv3)
      acd87(25)=spval5e1(iv3)
      acd87(26)=abb87(22)
      acd87(27)=spval5e1(iv2)
      acd87(28)=spval5e1(iv1)
      acd87(29)=acd87(3)*acd87(2)
      acd87(30)=acd87(6)*acd87(5)
      acd87(29)=acd87(29)+acd87(30)
      acd87(30)=acd87(1)*acd87(29)
      acd87(31)=acd87(8)*acd87(3)
      acd87(32)=acd87(9)*acd87(5)
      acd87(31)=acd87(31)+acd87(32)
      acd87(32)=acd87(7)*acd87(31)
      acd87(33)=acd87(8)*acd87(6)
      acd87(34)=acd87(9)*acd87(2)
      acd87(33)=acd87(33)+acd87(34)
      acd87(34)=acd87(10)*acd87(33)
      acd87(30)=acd87(34)+acd87(30)+acd87(32)
      acd87(30)=acd87(4)*acd87(30)
      acd87(32)=acd87(11)*acd87(33)
      acd87(31)=acd87(13)*acd87(31)
      acd87(29)=acd87(14)*acd87(29)
      acd87(29)=acd87(29)+acd87(31)+acd87(32)
      acd87(29)=acd87(12)*acd87(29)
      acd87(31)=acd87(17)*acd87(16)
      acd87(32)=acd87(20)*acd87(19)
      acd87(31)=acd87(31)+acd87(32)
      acd87(32)=acd87(15)*acd87(31)
      acd87(33)=acd87(22)*acd87(17)
      acd87(34)=acd87(23)*acd87(19)
      acd87(33)=acd87(33)+acd87(34)
      acd87(34)=acd87(21)*acd87(33)
      acd87(35)=acd87(22)*acd87(20)
      acd87(36)=acd87(23)*acd87(16)
      acd87(35)=acd87(35)+acd87(36)
      acd87(36)=acd87(24)*acd87(35)
      acd87(32)=acd87(36)+acd87(34)+acd87(32)
      acd87(32)=acd87(18)*acd87(32)
      acd87(34)=acd87(25)*acd87(35)
      acd87(33)=acd87(27)*acd87(33)
      acd87(31)=acd87(28)*acd87(31)
      acd87(31)=acd87(31)+acd87(33)+acd87(34)
      acd87(31)=acd87(26)*acd87(31)
      brack=acd87(29)+acd87(30)+acd87(31)+acd87(32)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd87h4_qp
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
      qshift = k2-k3-k5
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
end module     p2_gg_httbar_d87h4l1d_qp
