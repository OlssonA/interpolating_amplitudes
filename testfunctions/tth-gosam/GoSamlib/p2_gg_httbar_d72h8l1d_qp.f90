module     p2_gg_httbar_d72h8l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d72h8l1d_qp.f90
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
      use p2_gg_httbar_abbrevd72h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(59) :: acd72
      complex(ki) :: brack
      acd72(1)=dotproduct(qshift,qshift)
      acd72(2)=dotproduct(qshift,spvak1e2)
      acd72(3)=abb72(31)
      acd72(4)=dotproduct(qshift,spvae2k1)
      acd72(5)=abb72(23)
      acd72(6)=dotproduct(qshift,spvae2k2)
      acd72(7)=abb72(17)
      acd72(8)=dotproduct(qshift,spval4e2)
      acd72(9)=abb72(25)
      acd72(10)=dotproduct(qshift,spval5e2)
      acd72(11)=abb72(57)
      acd72(12)=dotproduct(qshift,spvae2l5)
      acd72(13)=abb72(47)
      acd72(14)=dotproduct(qshift,spvae1e2)
      acd72(15)=abb72(48)
      acd72(16)=dotproduct(qshift,spvae2e1)
      acd72(17)=abb72(40)
      acd72(18)=abb72(27)
      acd72(19)=abb72(28)
      acd72(20)=dotproduct(qshift,spvae2l3)
      acd72(21)=abb72(32)
      acd72(22)=abb72(30)
      acd72(23)=abb72(26)
      acd72(24)=dotproduct(qshift,spval3e2)
      acd72(25)=abb72(20)
      acd72(26)=abb72(13)
      acd72(27)=abb72(9)
      acd72(28)=abb72(10)
      acd72(29)=abb72(22)
      acd72(30)=abb72(11)
      acd72(31)=dotproduct(qshift,spvak2e2)
      acd72(32)=abb72(12)
      acd72(33)=abb72(15)
      acd72(34)=abb72(45)
      acd72(35)=abb72(29)
      acd72(36)=abb72(63)
      acd72(37)=abb72(19)
      acd72(38)=abb72(62)
      acd72(39)=abb72(55)
      acd72(40)=abb72(66)
      acd72(41)=abb72(24)
      acd72(42)=abb72(61)
      acd72(43)=abb72(39)
      acd72(44)=abb72(64)
      acd72(45)=abb72(16)
      acd72(46)=abb72(21)
      acd72(47)=abb72(18)
      acd72(48)=abb72(14)
      acd72(49)=-acd72(3)*acd72(2)
      acd72(50)=-acd72(5)*acd72(4)
      acd72(51)=-acd72(7)*acd72(6)
      acd72(52)=-acd72(9)*acd72(8)
      acd72(53)=-acd72(11)*acd72(10)
      acd72(54)=-acd72(13)*acd72(12)
      acd72(55)=-acd72(15)*acd72(14)
      acd72(56)=acd72(17)*acd72(16)
      acd72(49)=acd72(18)+acd72(56)+acd72(55)+acd72(54)+acd72(53)+acd72(52)+acd&
      &72(51)+acd72(49)+acd72(50)
      acd72(49)=acd72(1)*acd72(49)
      acd72(50)=acd72(19)*acd72(2)
      acd72(51)=acd72(27)*acd72(8)
      acd72(52)=acd72(28)*acd72(10)
      acd72(53)=acd72(29)*acd72(14)
      acd72(54)=acd72(30)*acd72(24)
      acd72(55)=acd72(32)*acd72(31)
      acd72(50)=-acd72(33)+acd72(55)+acd72(54)+acd72(53)+acd72(52)+acd72(51)+ac&
      &d72(50)
      acd72(50)=acd72(6)*acd72(50)
      acd72(51)=acd72(21)*acd72(2)
      acd72(52)=acd72(36)*acd72(8)
      acd72(53)=-acd72(38)*acd72(10)
      acd72(54)=acd72(42)*acd72(14)
      acd72(51)=-acd72(46)+acd72(54)+acd72(53)+acd72(52)+acd72(51)
      acd72(51)=acd72(20)*acd72(51)
      acd72(52)=acd72(23)*acd72(4)
      acd72(53)=-acd72(34)*acd72(12)
      acd72(54)=acd72(35)*acd72(16)
      acd72(52)=-acd72(37)+acd72(54)+acd72(53)+acd72(52)
      acd72(52)=acd72(8)*acd72(52)
      acd72(53)=acd72(25)*acd72(4)
      acd72(54)=acd72(40)*acd72(12)
      acd72(55)=acd72(44)*acd72(16)
      acd72(53)=-acd72(47)+acd72(55)+acd72(54)+acd72(53)
      acd72(53)=acd72(24)*acd72(53)
      acd72(54)=-acd72(22)*acd72(2)
      acd72(55)=-acd72(26)*acd72(4)
      acd72(56)=-acd72(39)*acd72(10)
      acd72(57)=-acd72(41)*acd72(12)
      acd72(58)=-acd72(43)*acd72(14)
      acd72(59)=-acd72(45)*acd72(16)
      brack=acd72(48)+acd72(49)+acd72(50)+acd72(51)+acd72(52)+acd72(53)+acd72(5&
      &4)+acd72(55)+acd72(56)+acd72(57)+acd72(58)+acd72(59)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd72h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(75) :: acd72
      complex(ki) :: brack
      acd72(1)=qshift(iv1)
      acd72(2)=dotproduct(qshift,spvak1e2)
      acd72(3)=abb72(31)
      acd72(4)=dotproduct(qshift,spvae2k1)
      acd72(5)=abb72(23)
      acd72(6)=dotproduct(qshift,spvae2k2)
      acd72(7)=abb72(17)
      acd72(8)=dotproduct(qshift,spval4e2)
      acd72(9)=abb72(25)
      acd72(10)=dotproduct(qshift,spval5e2)
      acd72(11)=abb72(57)
      acd72(12)=dotproduct(qshift,spvae2l5)
      acd72(13)=abb72(47)
      acd72(14)=dotproduct(qshift,spvae1e2)
      acd72(15)=abb72(48)
      acd72(16)=dotproduct(qshift,spvae2e1)
      acd72(17)=abb72(40)
      acd72(18)=abb72(27)
      acd72(19)=spvak1e2(iv1)
      acd72(20)=dotproduct(qshift,qshift)
      acd72(21)=abb72(28)
      acd72(22)=dotproduct(qshift,spvae2l3)
      acd72(23)=abb72(32)
      acd72(24)=abb72(30)
      acd72(25)=spvae2k1(iv1)
      acd72(26)=abb72(26)
      acd72(27)=dotproduct(qshift,spval3e2)
      acd72(28)=abb72(20)
      acd72(29)=abb72(13)
      acd72(30)=spvae2k2(iv1)
      acd72(31)=abb72(9)
      acd72(32)=abb72(10)
      acd72(33)=abb72(22)
      acd72(34)=abb72(11)
      acd72(35)=dotproduct(qshift,spvak2e2)
      acd72(36)=abb72(12)
      acd72(37)=abb72(15)
      acd72(38)=spval4e2(iv1)
      acd72(39)=abb72(45)
      acd72(40)=abb72(29)
      acd72(41)=abb72(63)
      acd72(42)=abb72(19)
      acd72(43)=spval5e2(iv1)
      acd72(44)=abb72(62)
      acd72(45)=abb72(55)
      acd72(46)=spvae2l5(iv1)
      acd72(47)=abb72(66)
      acd72(48)=abb72(24)
      acd72(49)=spvae1e2(iv1)
      acd72(50)=abb72(61)
      acd72(51)=abb72(39)
      acd72(52)=spvae2e1(iv1)
      acd72(53)=abb72(64)
      acd72(54)=abb72(16)
      acd72(55)=spvae2l3(iv1)
      acd72(56)=abb72(21)
      acd72(57)=spval3e2(iv1)
      acd72(58)=abb72(18)
      acd72(59)=spvak2e2(iv1)
      acd72(60)=-acd72(52)*acd72(17)
      acd72(61)=acd72(49)*acd72(15)
      acd72(62)=acd72(46)*acd72(13)
      acd72(63)=acd72(43)*acd72(11)
      acd72(64)=acd72(25)*acd72(5)
      acd72(65)=acd72(19)*acd72(3)
      acd72(66)=acd72(38)*acd72(9)
      acd72(67)=acd72(30)*acd72(7)
      acd72(60)=acd72(67)+acd72(66)+acd72(65)+acd72(64)+acd72(63)+acd72(62)+acd&
      &72(60)+acd72(61)
      acd72(60)=acd72(20)*acd72(60)
      acd72(61)=-acd72(16)*acd72(17)
      acd72(62)=acd72(14)*acd72(15)
      acd72(63)=acd72(12)*acd72(13)
      acd72(64)=acd72(10)*acd72(11)
      acd72(65)=acd72(4)*acd72(5)
      acd72(66)=acd72(2)*acd72(3)
      acd72(67)=acd72(8)*acd72(9)
      acd72(68)=acd72(6)*acd72(7)
      acd72(61)=acd72(68)+acd72(67)+acd72(66)+acd72(65)+acd72(64)+acd72(63)+acd&
      &72(62)-acd72(18)+acd72(61)
      acd72(61)=acd72(1)*acd72(61)
      acd72(62)=-acd72(36)*acd72(59)
      acd72(63)=-acd72(49)*acd72(33)
      acd72(64)=-acd72(43)*acd72(32)
      acd72(65)=-acd72(19)*acd72(21)
      acd72(66)=-acd72(57)*acd72(34)
      acd72(67)=-acd72(38)*acd72(31)
      acd72(62)=acd72(67)+acd72(66)+acd72(65)+acd72(64)+acd72(62)+acd72(63)
      acd72(62)=acd72(6)*acd72(62)
      acd72(63)=-acd72(36)*acd72(35)
      acd72(64)=-acd72(14)*acd72(33)
      acd72(65)=-acd72(10)*acd72(32)
      acd72(66)=-acd72(2)*acd72(21)
      acd72(67)=-acd72(27)*acd72(34)
      acd72(68)=-acd72(8)*acd72(31)
      acd72(63)=acd72(68)+acd72(67)+acd72(66)+acd72(65)+acd72(64)+acd72(37)+acd&
      &72(63)
      acd72(63)=acd72(30)*acd72(63)
      acd72(64)=-acd72(52)*acd72(40)
      acd72(65)=acd72(46)*acd72(39)
      acd72(66)=-acd72(25)*acd72(26)
      acd72(67)=-acd72(55)*acd72(41)
      acd72(64)=acd72(67)+acd72(66)+acd72(64)+acd72(65)
      acd72(64)=acd72(8)*acd72(64)
      acd72(65)=-acd72(16)*acd72(40)
      acd72(66)=acd72(12)*acd72(39)
      acd72(67)=-acd72(4)*acd72(26)
      acd72(68)=-acd72(22)*acd72(41)
      acd72(65)=acd72(68)+acd72(67)+acd72(66)+acd72(42)+acd72(65)
      acd72(65)=acd72(38)*acd72(65)
      acd72(66)=-acd72(16)*acd72(53)
      acd72(67)=-acd72(12)*acd72(47)
      acd72(68)=-acd72(4)*acd72(28)
      acd72(66)=acd72(68)+acd72(67)+acd72(58)+acd72(66)
      acd72(66)=acd72(57)*acd72(66)
      acd72(67)=-acd72(14)*acd72(50)
      acd72(68)=acd72(10)*acd72(44)
      acd72(69)=-acd72(2)*acd72(23)
      acd72(67)=acd72(69)+acd72(68)+acd72(56)+acd72(67)
      acd72(67)=acd72(55)*acd72(67)
      acd72(68)=-acd72(52)*acd72(53)
      acd72(69)=-acd72(46)*acd72(47)
      acd72(68)=acd72(68)+acd72(69)
      acd72(68)=acd72(27)*acd72(68)
      acd72(69)=-acd72(49)*acd72(50)
      acd72(70)=acd72(43)*acd72(44)
      acd72(69)=acd72(69)+acd72(70)
      acd72(69)=acd72(22)*acd72(69)
      acd72(70)=acd72(52)*acd72(54)
      acd72(71)=acd72(49)*acd72(51)
      acd72(72)=acd72(46)*acd72(48)
      acd72(73)=acd72(43)*acd72(45)
      acd72(74)=-acd72(27)*acd72(28)
      acd72(74)=acd72(29)+acd72(74)
      acd72(74)=acd72(25)*acd72(74)
      acd72(75)=-acd72(22)*acd72(23)
      acd72(75)=acd72(24)+acd72(75)
      acd72(75)=acd72(19)*acd72(75)
      brack=acd72(60)+2.0_ki*acd72(61)+acd72(62)+acd72(63)+acd72(64)+acd72(65)+&
      &acd72(66)+acd72(67)+acd72(68)+acd72(69)+acd72(70)+acd72(71)+acd72(72)+ac&
      &d72(73)+acd72(74)+acd72(75)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd72h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(69) :: acd72
      complex(ki) :: brack
      acd72(1)=d(iv1,iv2)
      acd72(2)=dotproduct(qshift,spvak1e2)
      acd72(3)=abb72(31)
      acd72(4)=dotproduct(qshift,spvae2k1)
      acd72(5)=abb72(23)
      acd72(6)=dotproduct(qshift,spvae2k2)
      acd72(7)=abb72(17)
      acd72(8)=dotproduct(qshift,spval4e2)
      acd72(9)=abb72(25)
      acd72(10)=dotproduct(qshift,spval5e2)
      acd72(11)=abb72(57)
      acd72(12)=dotproduct(qshift,spvae2l5)
      acd72(13)=abb72(47)
      acd72(14)=dotproduct(qshift,spvae1e2)
      acd72(15)=abb72(48)
      acd72(16)=dotproduct(qshift,spvae2e1)
      acd72(17)=abb72(40)
      acd72(18)=abb72(27)
      acd72(19)=qshift(iv1)
      acd72(20)=spvak1e2(iv2)
      acd72(21)=spvae2k1(iv2)
      acd72(22)=spvae2k2(iv2)
      acd72(23)=spval4e2(iv2)
      acd72(24)=spval5e2(iv2)
      acd72(25)=spvae2l5(iv2)
      acd72(26)=spvae1e2(iv2)
      acd72(27)=spvae2e1(iv2)
      acd72(28)=qshift(iv2)
      acd72(29)=spvak1e2(iv1)
      acd72(30)=spvae2k1(iv1)
      acd72(31)=spvae2k2(iv1)
      acd72(32)=spval4e2(iv1)
      acd72(33)=spval5e2(iv1)
      acd72(34)=spvae2l5(iv1)
      acd72(35)=spvae1e2(iv1)
      acd72(36)=spvae2e1(iv1)
      acd72(37)=abb72(28)
      acd72(38)=spvae2l3(iv2)
      acd72(39)=abb72(32)
      acd72(40)=spvae2l3(iv1)
      acd72(41)=abb72(26)
      acd72(42)=spval3e2(iv2)
      acd72(43)=abb72(20)
      acd72(44)=spval3e2(iv1)
      acd72(45)=abb72(9)
      acd72(46)=abb72(10)
      acd72(47)=abb72(22)
      acd72(48)=abb72(11)
      acd72(49)=spvak2e2(iv2)
      acd72(50)=abb72(12)
      acd72(51)=spvak2e2(iv1)
      acd72(52)=abb72(45)
      acd72(53)=abb72(29)
      acd72(54)=abb72(63)
      acd72(55)=abb72(62)
      acd72(56)=abb72(66)
      acd72(57)=abb72(61)
      acd72(58)=abb72(64)
      acd72(59)=acd72(17)*acd72(36)
      acd72(60)=-acd72(15)*acd72(35)
      acd72(61)=-acd72(13)*acd72(34)
      acd72(62)=-acd72(11)*acd72(33)
      acd72(63)=-acd72(5)*acd72(30)
      acd72(64)=-acd72(3)*acd72(29)
      acd72(65)=-acd72(32)*acd72(9)
      acd72(66)=-acd72(31)*acd72(7)
      acd72(59)=acd72(66)+acd72(65)+acd72(64)+acd72(63)+acd72(62)+acd72(61)+acd&
      &72(59)+acd72(60)
      acd72(59)=acd72(28)*acd72(59)
      acd72(60)=acd72(17)*acd72(27)
      acd72(61)=-acd72(15)*acd72(26)
      acd72(62)=-acd72(13)*acd72(25)
      acd72(63)=-acd72(11)*acd72(24)
      acd72(64)=-acd72(5)*acd72(21)
      acd72(65)=-acd72(3)*acd72(20)
      acd72(66)=-acd72(23)*acd72(9)
      acd72(67)=-acd72(22)*acd72(7)
      acd72(60)=acd72(67)+acd72(66)+acd72(65)+acd72(64)+acd72(63)+acd72(62)+acd&
      &72(60)+acd72(61)
      acd72(60)=acd72(19)*acd72(60)
      acd72(61)=acd72(17)*acd72(16)
      acd72(62)=-acd72(15)*acd72(14)
      acd72(63)=-acd72(13)*acd72(12)
      acd72(64)=-acd72(11)*acd72(10)
      acd72(65)=-acd72(9)*acd72(8)
      acd72(66)=-acd72(7)*acd72(6)
      acd72(67)=-acd72(5)*acd72(4)
      acd72(68)=-acd72(3)*acd72(2)
      acd72(61)=acd72(68)+acd72(67)+acd72(66)+acd72(65)+acd72(64)+acd72(63)+acd&
      &72(62)+acd72(18)+acd72(61)
      acd72(61)=acd72(1)*acd72(61)
      acd72(59)=acd72(61)+acd72(59)+acd72(60)
      acd72(60)=acd72(50)*acd72(49)
      acd72(61)=acd72(26)*acd72(47)
      acd72(62)=acd72(24)*acd72(46)
      acd72(63)=acd72(20)*acd72(37)
      acd72(64)=acd72(42)*acd72(48)
      acd72(65)=acd72(23)*acd72(45)
      acd72(60)=acd72(65)+acd72(64)+acd72(63)+acd72(62)+acd72(60)+acd72(61)
      acd72(60)=acd72(31)*acd72(60)
      acd72(61)=acd72(50)*acd72(51)
      acd72(62)=acd72(35)*acd72(47)
      acd72(63)=acd72(33)*acd72(46)
      acd72(64)=acd72(29)*acd72(37)
      acd72(65)=acd72(44)*acd72(48)
      acd72(66)=acd72(32)*acd72(45)
      acd72(61)=acd72(66)+acd72(65)+acd72(64)+acd72(63)+acd72(61)+acd72(62)
      acd72(61)=acd72(22)*acd72(61)
      acd72(62)=acd72(27)*acd72(53)
      acd72(63)=-acd72(25)*acd72(52)
      acd72(64)=acd72(21)*acd72(41)
      acd72(65)=acd72(38)*acd72(54)
      acd72(62)=acd72(65)+acd72(64)+acd72(62)+acd72(63)
      acd72(62)=acd72(32)*acd72(62)
      acd72(63)=acd72(36)*acd72(53)
      acd72(64)=-acd72(34)*acd72(52)
      acd72(65)=acd72(30)*acd72(41)
      acd72(66)=acd72(40)*acd72(54)
      acd72(63)=acd72(66)+acd72(65)+acd72(63)+acd72(64)
      acd72(63)=acd72(23)*acd72(63)
      acd72(64)=acd72(27)*acd72(58)
      acd72(65)=acd72(25)*acd72(56)
      acd72(66)=acd72(21)*acd72(43)
      acd72(64)=acd72(66)+acd72(64)+acd72(65)
      acd72(64)=acd72(44)*acd72(64)
      acd72(65)=acd72(36)*acd72(58)
      acd72(66)=acd72(34)*acd72(56)
      acd72(67)=acd72(30)*acd72(43)
      acd72(65)=acd72(67)+acd72(65)+acd72(66)
      acd72(65)=acd72(42)*acd72(65)
      acd72(66)=acd72(26)*acd72(57)
      acd72(67)=-acd72(24)*acd72(55)
      acd72(68)=acd72(20)*acd72(39)
      acd72(66)=acd72(68)+acd72(66)+acd72(67)
      acd72(66)=acd72(40)*acd72(66)
      acd72(67)=acd72(35)*acd72(57)
      acd72(68)=-acd72(33)*acd72(55)
      acd72(69)=acd72(29)*acd72(39)
      acd72(67)=acd72(69)+acd72(67)+acd72(68)
      acd72(67)=acd72(38)*acd72(67)
      brack=2.0_ki*acd72(59)+acd72(60)+acd72(61)+acd72(62)+acd72(63)+acd72(64)+&
      &acd72(65)+acd72(66)+acd72(67)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd72h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd72
      complex(ki) :: brack
      acd72(1)=d(iv1,iv2)
      acd72(2)=spvak1e2(iv3)
      acd72(3)=abb72(31)
      acd72(4)=spvae2k1(iv3)
      acd72(5)=abb72(23)
      acd72(6)=spvae2k2(iv3)
      acd72(7)=abb72(17)
      acd72(8)=spval4e2(iv3)
      acd72(9)=abb72(25)
      acd72(10)=spval5e2(iv3)
      acd72(11)=abb72(57)
      acd72(12)=spvae2l5(iv3)
      acd72(13)=abb72(47)
      acd72(14)=spvae1e2(iv3)
      acd72(15)=abb72(48)
      acd72(16)=spvae2e1(iv3)
      acd72(17)=abb72(40)
      acd72(18)=d(iv1,iv3)
      acd72(19)=spvak1e2(iv2)
      acd72(20)=spvae2k1(iv2)
      acd72(21)=spvae2k2(iv2)
      acd72(22)=spval4e2(iv2)
      acd72(23)=spval5e2(iv2)
      acd72(24)=spvae2l5(iv2)
      acd72(25)=spvae1e2(iv2)
      acd72(26)=spvae2e1(iv2)
      acd72(27)=d(iv2,iv3)
      acd72(28)=spvak1e2(iv1)
      acd72(29)=spvae2k1(iv1)
      acd72(30)=spvae2k2(iv1)
      acd72(31)=spval4e2(iv1)
      acd72(32)=spval5e2(iv1)
      acd72(33)=spvae2l5(iv1)
      acd72(34)=spvae1e2(iv1)
      acd72(35)=spvae2e1(iv1)
      acd72(36)=acd72(2)*acd72(3)
      acd72(37)=acd72(4)*acd72(5)
      acd72(38)=acd72(6)*acd72(7)
      acd72(39)=acd72(8)*acd72(9)
      acd72(40)=acd72(10)*acd72(11)
      acd72(41)=acd72(12)*acd72(13)
      acd72(42)=acd72(14)*acd72(15)
      acd72(43)=-acd72(16)*acd72(17)
      acd72(36)=acd72(43)+acd72(42)+acd72(41)+acd72(40)+acd72(39)+acd72(38)+acd&
      &72(36)+acd72(37)
      acd72(36)=acd72(1)*acd72(36)
      acd72(37)=acd72(19)*acd72(3)
      acd72(38)=acd72(20)*acd72(5)
      acd72(39)=acd72(21)*acd72(7)
      acd72(40)=acd72(22)*acd72(9)
      acd72(41)=acd72(23)*acd72(11)
      acd72(42)=acd72(24)*acd72(13)
      acd72(43)=acd72(25)*acd72(15)
      acd72(44)=-acd72(26)*acd72(17)
      acd72(37)=acd72(44)+acd72(43)+acd72(42)+acd72(41)+acd72(40)+acd72(39)+acd&
      &72(38)+acd72(37)
      acd72(37)=acd72(18)*acd72(37)
      acd72(38)=acd72(28)*acd72(3)
      acd72(39)=acd72(29)*acd72(5)
      acd72(40)=acd72(30)*acd72(7)
      acd72(41)=acd72(31)*acd72(9)
      acd72(42)=acd72(32)*acd72(11)
      acd72(43)=acd72(33)*acd72(13)
      acd72(44)=acd72(34)*acd72(15)
      acd72(45)=-acd72(35)*acd72(17)
      acd72(38)=acd72(45)+acd72(44)+acd72(43)+acd72(42)+acd72(41)+acd72(40)+acd&
      &72(39)+acd72(38)
      acd72(38)=acd72(27)*acd72(38)
      acd72(36)=acd72(38)+acd72(37)+acd72(36)
      brack=2.0_ki*acd72(36)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd72h8_qp
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
      qshift = -k2+k4
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
end module     p2_gg_httbar_d72h8l1d_qp
