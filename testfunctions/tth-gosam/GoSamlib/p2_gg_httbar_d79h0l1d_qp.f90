module     p2_gg_httbar_d79h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d79h0l1d_qp.f90
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
      use p2_gg_httbar_abbrevd79h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(63) :: acd79
      complex(ki) :: brack
      acd79(1)=dotproduct(k2,qshift)
      acd79(2)=dotproduct(qshift,spvak2e1)
      acd79(3)=abb79(20)
      acd79(4)=dotproduct(qshift,spvae1k2)
      acd79(5)=abb79(23)
      acd79(6)=dotproduct(qshift,qshift)
      acd79(7)=abb79(16)
      acd79(8)=abb79(33)
      acd79(9)=dotproduct(qshift,spval4e1)
      acd79(10)=abb79(28)
      acd79(11)=dotproduct(qshift,spval5e1)
      acd79(12)=abb79(38)
      acd79(13)=dotproduct(qshift,spvae1l5)
      acd79(14)=abb79(32)
      acd79(15)=dotproduct(qshift,spvae1e2)
      acd79(16)=abb79(55)
      acd79(17)=dotproduct(qshift,spvae2e1)
      acd79(18)=abb79(15)
      acd79(19)=abb79(14)
      acd79(20)=abb79(24)
      acd79(21)=dotproduct(qshift,spval4k2)
      acd79(22)=abb79(9)
      acd79(23)=dotproduct(qshift,spval5k2)
      acd79(24)=abb79(35)
      acd79(25)=dotproduct(qshift,spvae2k2)
      acd79(26)=abb79(27)
      acd79(27)=abb79(18)
      acd79(28)=abb79(31)
      acd79(29)=abb79(25)
      acd79(30)=dotproduct(qshift,spvak2l5)
      acd79(31)=abb79(10)
      acd79(32)=dotproduct(qshift,spvak2e2)
      acd79(33)=abb79(59)
      acd79(34)=abb79(19)
      acd79(35)=abb79(12)
      acd79(36)=abb79(30)
      acd79(37)=abb79(17)
      acd79(38)=abb79(34)
      acd79(39)=abb79(29)
      acd79(40)=abb79(26)
      acd79(41)=abb79(22)
      acd79(42)=dotproduct(qshift,spvak2l3)
      acd79(43)=dotproduct(qshift,spval3e1)
      acd79(44)=dotproduct(qshift,spval4l3)
      acd79(45)=dotproduct(qshift,spval5l3)
      acd79(46)=dotproduct(qshift,spvae2l3)
      acd79(47)=abb79(11)
      acd79(48)=dotproduct(qshift,spval3k2)
      acd79(49)=dotproduct(qshift,spvae1l3)
      acd79(50)=dotproduct(qshift,spval3l5)
      acd79(51)=dotproduct(qshift,spval3e2)
      acd79(52)=abb79(21)
      acd79(53)=abb79(13)
      acd79(54)=-acd79(7)*acd79(2)
      acd79(55)=-acd79(8)*acd79(4)
      acd79(56)=-acd79(10)*acd79(9)
      acd79(57)=-acd79(12)*acd79(11)
      acd79(58)=acd79(14)*acd79(13)
      acd79(59)=acd79(16)*acd79(15)
      acd79(60)=acd79(18)*acd79(17)
      acd79(54)=acd79(19)+acd79(60)+acd79(59)+acd79(58)+acd79(57)+acd79(56)+acd&
      &79(54)+acd79(55)
      acd79(54)=acd79(6)*acd79(54)
      acd79(55)=acd79(5)*acd79(1)
      acd79(56)=acd79(20)*acd79(2)
      acd79(57)=acd79(28)*acd79(11)
      acd79(58)=acd79(29)*acd79(17)
      acd79(59)=acd79(31)*acd79(30)
      acd79(60)=-acd79(33)*acd79(32)
      acd79(55)=-acd79(34)+acd79(60)+acd79(59)+acd79(58)+acd79(57)+acd79(56)+ac&
      &d79(55)
      acd79(55)=acd79(4)*acd79(55)
      acd79(56)=acd79(3)*acd79(1)
      acd79(57)=acd79(22)*acd79(21)
      acd79(58)=acd79(24)*acd79(23)
      acd79(59)=acd79(26)*acd79(25)
      acd79(56)=-acd79(27)+acd79(59)+acd79(58)+acd79(57)+acd79(56)
      acd79(56)=acd79(2)*acd79(56)
      acd79(57)=acd79(42)*acd79(7)
      acd79(58)=acd79(44)*acd79(10)
      acd79(59)=acd79(45)*acd79(12)
      acd79(60)=-acd79(46)*acd79(18)
      acd79(57)=-acd79(47)+acd79(60)+acd79(59)+acd79(58)+acd79(57)
      acd79(57)=acd79(43)*acd79(57)
      acd79(58)=acd79(48)*acd79(8)
      acd79(59)=-acd79(50)*acd79(14)
      acd79(60)=-acd79(51)*acd79(16)
      acd79(58)=-acd79(52)+acd79(60)+acd79(59)+acd79(58)
      acd79(58)=acd79(49)*acd79(58)
      acd79(59)=acd79(35)*acd79(13)
      acd79(60)=acd79(36)*acd79(15)
      acd79(59)=-acd79(37)+acd79(60)+acd79(59)
      acd79(59)=acd79(9)*acd79(59)
      acd79(60)=-acd79(38)*acd79(11)
      acd79(61)=-acd79(39)*acd79(13)
      acd79(62)=-acd79(40)*acd79(15)
      acd79(63)=-acd79(41)*acd79(17)
      brack=acd79(53)+acd79(54)+acd79(55)+acd79(56)+acd79(57)+acd79(58)+acd79(5&
      &9)+acd79(60)+acd79(61)+acd79(62)+acd79(63)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd79h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(93) :: acd79
      complex(ki) :: brack
      acd79(1)=k2(iv1)
      acd79(2)=dotproduct(qshift,spvak2e1)
      acd79(3)=abb79(20)
      acd79(4)=dotproduct(qshift,spvae1k2)
      acd79(5)=abb79(23)
      acd79(6)=qshift(iv1)
      acd79(7)=abb79(16)
      acd79(8)=abb79(33)
      acd79(9)=dotproduct(qshift,spval4e1)
      acd79(10)=abb79(28)
      acd79(11)=dotproduct(qshift,spval5e1)
      acd79(12)=abb79(38)
      acd79(13)=dotproduct(qshift,spvae1l5)
      acd79(14)=abb79(32)
      acd79(15)=dotproduct(qshift,spvae1e2)
      acd79(16)=abb79(55)
      acd79(17)=dotproduct(qshift,spvae2e1)
      acd79(18)=abb79(15)
      acd79(19)=abb79(14)
      acd79(20)=spvak2e1(iv1)
      acd79(21)=dotproduct(k2,qshift)
      acd79(22)=dotproduct(qshift,qshift)
      acd79(23)=abb79(24)
      acd79(24)=dotproduct(qshift,spval4k2)
      acd79(25)=abb79(9)
      acd79(26)=dotproduct(qshift,spval5k2)
      acd79(27)=abb79(35)
      acd79(28)=dotproduct(qshift,spvae2k2)
      acd79(29)=abb79(27)
      acd79(30)=abb79(18)
      acd79(31)=spvae1k2(iv1)
      acd79(32)=abb79(31)
      acd79(33)=abb79(25)
      acd79(34)=dotproduct(qshift,spvak2l5)
      acd79(35)=abb79(10)
      acd79(36)=dotproduct(qshift,spvak2e2)
      acd79(37)=abb79(59)
      acd79(38)=abb79(19)
      acd79(39)=spval4e1(iv1)
      acd79(40)=abb79(12)
      acd79(41)=abb79(30)
      acd79(42)=abb79(17)
      acd79(43)=spval5e1(iv1)
      acd79(44)=abb79(34)
      acd79(45)=spvae1l5(iv1)
      acd79(46)=abb79(29)
      acd79(47)=spvae1e2(iv1)
      acd79(48)=abb79(26)
      acd79(49)=spvae2e1(iv1)
      acd79(50)=abb79(22)
      acd79(51)=spvak2l3(iv1)
      acd79(52)=dotproduct(qshift,spval3e1)
      acd79(53)=spval3e1(iv1)
      acd79(54)=dotproduct(qshift,spvak2l3)
      acd79(55)=dotproduct(qshift,spval4l3)
      acd79(56)=dotproduct(qshift,spval5l3)
      acd79(57)=dotproduct(qshift,spvae2l3)
      acd79(58)=abb79(11)
      acd79(59)=spvak2l5(iv1)
      acd79(60)=spval3k2(iv1)
      acd79(61)=dotproduct(qshift,spvae1l3)
      acd79(62)=spvae1l3(iv1)
      acd79(63)=dotproduct(qshift,spval3k2)
      acd79(64)=dotproduct(qshift,spval3l5)
      acd79(65)=dotproduct(qshift,spval3e2)
      acd79(66)=abb79(21)
      acd79(67)=spval3l5(iv1)
      acd79(68)=spval4k2(iv1)
      acd79(69)=spval4l3(iv1)
      acd79(70)=spval5k2(iv1)
      acd79(71)=spval5l3(iv1)
      acd79(72)=spvae2k2(iv1)
      acd79(73)=spvak2e2(iv1)
      acd79(74)=spvae2l3(iv1)
      acd79(75)=spval3e2(iv1)
      acd79(76)=acd79(37)*acd79(36)
      acd79(77)=-acd79(35)*acd79(34)
      acd79(78)=-acd79(17)*acd79(33)
      acd79(79)=-acd79(11)*acd79(32)
      acd79(80)=-acd79(5)*acd79(21)
      acd79(81)=-acd79(2)*acd79(23)
      acd79(82)=acd79(22)*acd79(8)
      acd79(76)=acd79(82)+acd79(81)+acd79(80)+acd79(79)+acd79(78)+acd79(77)+acd&
      &79(38)+acd79(76)
      acd79(76)=acd79(31)*acd79(76)
      acd79(77)=-acd79(18)*acd79(17)
      acd79(78)=-acd79(16)*acd79(15)
      acd79(79)=-acd79(14)*acd79(13)
      acd79(80)=acd79(12)*acd79(11)
      acd79(81)=acd79(10)*acd79(9)
      acd79(82)=acd79(2)*acd79(7)
      acd79(83)=acd79(4)*acd79(8)
      acd79(77)=acd79(83)+acd79(82)+acd79(81)+acd79(80)+acd79(79)+acd79(78)-acd&
      &79(19)+acd79(77)
      acd79(77)=acd79(6)*acd79(77)
      acd79(78)=acd79(37)*acd79(73)
      acd79(79)=-acd79(35)*acd79(59)
      acd79(80)=-acd79(1)*acd79(5)
      acd79(81)=-acd79(49)*acd79(33)
      acd79(82)=-acd79(43)*acd79(32)
      acd79(83)=-acd79(20)*acd79(23)
      acd79(78)=acd79(83)+acd79(82)+acd79(81)+acd79(80)+acd79(78)+acd79(79)
      acd79(78)=acd79(4)*acd79(78)
      acd79(79)=-acd79(18)*acd79(49)
      acd79(80)=-acd79(16)*acd79(47)
      acd79(81)=-acd79(14)*acd79(45)
      acd79(82)=acd79(12)*acd79(43)
      acd79(83)=acd79(10)*acd79(39)
      acd79(79)=acd79(83)+acd79(82)+acd79(81)+acd79(79)+acd79(80)
      acd79(79)=acd79(22)*acd79(79)
      acd79(80)=-acd79(29)*acd79(28)
      acd79(81)=-acd79(27)*acd79(26)
      acd79(82)=-acd79(25)*acd79(24)
      acd79(83)=-acd79(3)*acd79(21)
      acd79(84)=acd79(22)*acd79(7)
      acd79(80)=acd79(84)+acd79(83)+acd79(82)+acd79(81)+acd79(30)+acd79(80)
      acd79(80)=acd79(20)*acd79(80)
      acd79(81)=acd79(18)*acd79(74)
      acd79(82)=-acd79(12)*acd79(71)
      acd79(83)=-acd79(10)*acd79(69)
      acd79(84)=-acd79(7)*acd79(51)
      acd79(81)=acd79(84)+acd79(83)+acd79(82)+acd79(81)
      acd79(81)=acd79(52)*acd79(81)
      acd79(82)=acd79(18)*acd79(57)
      acd79(83)=-acd79(12)*acd79(56)
      acd79(84)=-acd79(10)*acd79(55)
      acd79(85)=-acd79(7)*acd79(54)
      acd79(82)=acd79(85)+acd79(84)+acd79(83)+acd79(58)+acd79(82)
      acd79(82)=acd79(53)*acd79(82)
      acd79(83)=-acd79(29)*acd79(72)
      acd79(84)=-acd79(27)*acd79(70)
      acd79(85)=-acd79(25)*acd79(68)
      acd79(86)=-acd79(1)*acd79(3)
      acd79(83)=acd79(86)+acd79(85)+acd79(83)+acd79(84)
      acd79(83)=acd79(2)*acd79(83)
      acd79(84)=-acd79(47)*acd79(41)
      acd79(85)=-acd79(45)*acd79(40)
      acd79(84)=acd79(84)+acd79(85)
      acd79(84)=acd79(9)*acd79(84)
      acd79(85)=-acd79(15)*acd79(41)
      acd79(86)=-acd79(13)*acd79(40)
      acd79(85)=acd79(86)+acd79(42)+acd79(85)
      acd79(85)=acd79(39)*acd79(85)
      acd79(86)=acd79(61)*acd79(75)
      acd79(87)=acd79(62)*acd79(65)
      acd79(86)=acd79(86)+acd79(87)
      acd79(86)=acd79(16)*acd79(86)
      acd79(87)=acd79(61)*acd79(67)
      acd79(88)=acd79(62)*acd79(64)
      acd79(87)=acd79(87)+acd79(88)
      acd79(87)=acd79(14)*acd79(87)
      acd79(88)=-acd79(61)*acd79(60)
      acd79(89)=-acd79(62)*acd79(63)
      acd79(88)=acd79(88)+acd79(89)
      acd79(88)=acd79(8)*acd79(88)
      acd79(89)=acd79(49)*acd79(50)
      acd79(90)=acd79(47)*acd79(48)
      acd79(91)=acd79(45)*acd79(46)
      acd79(92)=acd79(43)*acd79(44)
      acd79(93)=acd79(62)*acd79(66)
      brack=acd79(76)+2.0_ki*acd79(77)+acd79(78)+acd79(79)+acd79(80)+acd79(81)+&
      &acd79(82)+acd79(83)+acd79(84)+acd79(85)+acd79(86)+acd79(87)+acd79(88)+ac&
      &d79(89)+acd79(90)+acd79(91)+acd79(92)+acd79(93)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd79h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(91) :: acd79
      complex(ki) :: brack
      acd79(1)=d(iv1,iv2)
      acd79(2)=dotproduct(qshift,spvak2e1)
      acd79(3)=abb79(16)
      acd79(4)=dotproduct(qshift,spvae1k2)
      acd79(5)=abb79(33)
      acd79(6)=dotproduct(qshift,spval4e1)
      acd79(7)=abb79(28)
      acd79(8)=dotproduct(qshift,spval5e1)
      acd79(9)=abb79(38)
      acd79(10)=dotproduct(qshift,spvae1l5)
      acd79(11)=abb79(32)
      acd79(12)=dotproduct(qshift,spvae1e2)
      acd79(13)=abb79(55)
      acd79(14)=dotproduct(qshift,spvae2e1)
      acd79(15)=abb79(15)
      acd79(16)=abb79(14)
      acd79(17)=k2(iv1)
      acd79(18)=spvak2e1(iv2)
      acd79(19)=abb79(20)
      acd79(20)=spvae1k2(iv2)
      acd79(21)=abb79(23)
      acd79(22)=k2(iv2)
      acd79(23)=spvak2e1(iv1)
      acd79(24)=spvae1k2(iv1)
      acd79(25)=qshift(iv1)
      acd79(26)=spval4e1(iv2)
      acd79(27)=spval5e1(iv2)
      acd79(28)=spvae1l5(iv2)
      acd79(29)=spvae1e2(iv2)
      acd79(30)=spvae2e1(iv2)
      acd79(31)=qshift(iv2)
      acd79(32)=spval4e1(iv1)
      acd79(33)=spval5e1(iv1)
      acd79(34)=spvae1l5(iv1)
      acd79(35)=spvae1e2(iv1)
      acd79(36)=spvae2e1(iv1)
      acd79(37)=abb79(24)
      acd79(38)=spval4k2(iv2)
      acd79(39)=abb79(9)
      acd79(40)=spval5k2(iv2)
      acd79(41)=abb79(35)
      acd79(42)=spvae2k2(iv2)
      acd79(43)=abb79(27)
      acd79(44)=spval4k2(iv1)
      acd79(45)=spval5k2(iv1)
      acd79(46)=spvae2k2(iv1)
      acd79(47)=abb79(31)
      acd79(48)=abb79(25)
      acd79(49)=spvak2l5(iv2)
      acd79(50)=abb79(10)
      acd79(51)=spvak2e2(iv2)
      acd79(52)=abb79(59)
      acd79(53)=spvak2l5(iv1)
      acd79(54)=spvak2e2(iv1)
      acd79(55)=abb79(12)
      acd79(56)=abb79(30)
      acd79(57)=spvak2l3(iv1)
      acd79(58)=spval3e1(iv2)
      acd79(59)=spvak2l3(iv2)
      acd79(60)=spval3e1(iv1)
      acd79(61)=spval4l3(iv2)
      acd79(62)=spval5l3(iv2)
      acd79(63)=spvae2l3(iv2)
      acd79(64)=spval4l3(iv1)
      acd79(65)=spval5l3(iv1)
      acd79(66)=spvae2l3(iv1)
      acd79(67)=spval3k2(iv1)
      acd79(68)=spvae1l3(iv2)
      acd79(69)=spval3k2(iv2)
      acd79(70)=spvae1l3(iv1)
      acd79(71)=spval3l5(iv2)
      acd79(72)=spval3e2(iv2)
      acd79(73)=spval3l5(iv1)
      acd79(74)=spval3e2(iv1)
      acd79(75)=-acd79(52)*acd79(51)
      acd79(76)=acd79(50)*acd79(49)
      acd79(77)=acd79(30)*acd79(48)
      acd79(78)=acd79(27)*acd79(47)
      acd79(79)=acd79(21)*acd79(22)
      acd79(80)=acd79(18)*acd79(37)
      acd79(81)=2.0_ki*acd79(31)
      acd79(82)=-acd79(5)*acd79(81)
      acd79(75)=acd79(82)+acd79(80)+acd79(79)+acd79(78)+acd79(77)+acd79(75)+acd&
      &79(76)
      acd79(75)=acd79(24)*acd79(75)
      acd79(76)=-acd79(52)*acd79(54)
      acd79(77)=acd79(50)*acd79(53)
      acd79(78)=acd79(36)*acd79(48)
      acd79(79)=acd79(33)*acd79(47)
      acd79(80)=acd79(17)*acd79(21)
      acd79(82)=acd79(23)*acd79(37)
      acd79(83)=2.0_ki*acd79(25)
      acd79(84)=-acd79(5)*acd79(83)
      acd79(76)=acd79(84)+acd79(82)+acd79(80)+acd79(79)+acd79(78)+acd79(76)+acd&
      &79(77)
      acd79(76)=acd79(20)*acd79(76)
      acd79(77)=acd79(15)*acd79(14)
      acd79(78)=acd79(13)*acd79(12)
      acd79(79)=acd79(11)*acd79(10)
      acd79(80)=-acd79(9)*acd79(8)
      acd79(82)=-acd79(7)*acd79(6)
      acd79(84)=-acd79(5)*acd79(4)
      acd79(85)=-acd79(3)*acd79(2)
      acd79(77)=acd79(85)+acd79(84)+acd79(82)+acd79(80)+acd79(79)+acd79(78)+acd&
      &79(16)+acd79(77)
      acd79(77)=acd79(1)*acd79(77)
      acd79(78)=acd79(15)*acd79(36)
      acd79(79)=acd79(13)*acd79(35)
      acd79(80)=acd79(11)*acd79(34)
      acd79(82)=-acd79(9)*acd79(33)
      acd79(84)=-acd79(7)*acd79(32)
      acd79(85)=-acd79(23)*acd79(3)
      acd79(78)=acd79(85)+acd79(84)+acd79(82)+acd79(80)+acd79(78)+acd79(79)
      acd79(78)=acd79(78)*acd79(81)
      acd79(79)=acd79(15)*acd79(30)
      acd79(80)=acd79(13)*acd79(29)
      acd79(81)=acd79(11)*acd79(28)
      acd79(82)=-acd79(9)*acd79(27)
      acd79(84)=-acd79(7)*acd79(26)
      acd79(85)=-acd79(18)*acd79(3)
      acd79(79)=acd79(85)+acd79(84)+acd79(82)+acd79(81)+acd79(79)+acd79(80)
      acd79(79)=acd79(79)*acd79(83)
      acd79(80)=acd79(43)*acd79(42)
      acd79(81)=acd79(41)*acd79(40)
      acd79(82)=acd79(39)*acd79(38)
      acd79(83)=acd79(19)*acd79(22)
      acd79(80)=acd79(83)+acd79(82)+acd79(80)+acd79(81)
      acd79(80)=acd79(23)*acd79(80)
      acd79(81)=acd79(43)*acd79(46)
      acd79(82)=acd79(41)*acd79(45)
      acd79(83)=acd79(39)*acd79(44)
      acd79(84)=acd79(17)*acd79(19)
      acd79(81)=acd79(84)+acd79(83)+acd79(81)+acd79(82)
      acd79(81)=acd79(18)*acd79(81)
      acd79(82)=acd79(29)*acd79(56)
      acd79(83)=acd79(28)*acd79(55)
      acd79(82)=acd79(82)+acd79(83)
      acd79(82)=acd79(32)*acd79(82)
      acd79(83)=acd79(35)*acd79(56)
      acd79(84)=acd79(34)*acd79(55)
      acd79(83)=acd79(83)+acd79(84)
      acd79(83)=acd79(26)*acd79(83)
      acd79(84)=-acd79(60)*acd79(63)
      acd79(85)=-acd79(58)*acd79(66)
      acd79(84)=acd79(84)+acd79(85)
      acd79(84)=acd79(15)*acd79(84)
      acd79(85)=-acd79(70)*acd79(72)
      acd79(86)=-acd79(68)*acd79(74)
      acd79(85)=acd79(85)+acd79(86)
      acd79(85)=acd79(13)*acd79(85)
      acd79(86)=-acd79(70)*acd79(71)
      acd79(87)=-acd79(68)*acd79(73)
      acd79(86)=acd79(86)+acd79(87)
      acd79(86)=acd79(11)*acd79(86)
      acd79(87)=acd79(60)*acd79(62)
      acd79(88)=acd79(58)*acd79(65)
      acd79(87)=acd79(87)+acd79(88)
      acd79(87)=acd79(9)*acd79(87)
      acd79(88)=acd79(60)*acd79(61)
      acd79(89)=acd79(58)*acd79(64)
      acd79(88)=acd79(88)+acd79(89)
      acd79(88)=acd79(7)*acd79(88)
      acd79(89)=acd79(70)*acd79(69)
      acd79(90)=acd79(68)*acd79(67)
      acd79(89)=acd79(89)+acd79(90)
      acd79(89)=acd79(5)*acd79(89)
      acd79(90)=acd79(60)*acd79(59)
      acd79(91)=acd79(58)*acd79(57)
      acd79(90)=acd79(90)+acd79(91)
      acd79(90)=acd79(3)*acd79(90)
      brack=acd79(75)+acd79(76)+2.0_ki*acd79(77)+acd79(78)+acd79(79)+acd79(80)+&
      &acd79(81)+acd79(82)+acd79(83)+acd79(84)+acd79(85)+acd79(86)+acd79(87)+ac&
      &d79(88)+acd79(89)+acd79(90)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd79h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd79
      complex(ki) :: brack
      acd79(1)=d(iv1,iv2)
      acd79(2)=spvak2e1(iv3)
      acd79(3)=abb79(16)
      acd79(4)=spvae1k2(iv3)
      acd79(5)=abb79(33)
      acd79(6)=spval4e1(iv3)
      acd79(7)=abb79(28)
      acd79(8)=spval5e1(iv3)
      acd79(9)=abb79(38)
      acd79(10)=spvae1l5(iv3)
      acd79(11)=abb79(32)
      acd79(12)=spvae1e2(iv3)
      acd79(13)=abb79(55)
      acd79(14)=spvae2e1(iv3)
      acd79(15)=abb79(15)
      acd79(16)=d(iv1,iv3)
      acd79(17)=spvak2e1(iv2)
      acd79(18)=spvae1k2(iv2)
      acd79(19)=spval4e1(iv2)
      acd79(20)=spval5e1(iv2)
      acd79(21)=spvae1l5(iv2)
      acd79(22)=spvae1e2(iv2)
      acd79(23)=spvae2e1(iv2)
      acd79(24)=d(iv2,iv3)
      acd79(25)=spvak2e1(iv1)
      acd79(26)=spvae1k2(iv1)
      acd79(27)=spval4e1(iv1)
      acd79(28)=spval5e1(iv1)
      acd79(29)=spvae1l5(iv1)
      acd79(30)=spvae1e2(iv1)
      acd79(31)=spvae2e1(iv1)
      acd79(32)=acd79(2)*acd79(3)
      acd79(33)=acd79(4)*acd79(5)
      acd79(34)=acd79(6)*acd79(7)
      acd79(35)=acd79(8)*acd79(9)
      acd79(36)=-acd79(10)*acd79(11)
      acd79(37)=-acd79(12)*acd79(13)
      acd79(38)=-acd79(14)*acd79(15)
      acd79(32)=acd79(38)+acd79(37)+acd79(36)+acd79(35)+acd79(34)+acd79(32)+acd&
      &79(33)
      acd79(32)=acd79(1)*acd79(32)
      acd79(33)=acd79(17)*acd79(3)
      acd79(34)=acd79(18)*acd79(5)
      acd79(35)=acd79(19)*acd79(7)
      acd79(36)=acd79(20)*acd79(9)
      acd79(37)=-acd79(21)*acd79(11)
      acd79(38)=-acd79(22)*acd79(13)
      acd79(39)=-acd79(23)*acd79(15)
      acd79(33)=acd79(39)+acd79(38)+acd79(37)+acd79(36)+acd79(35)+acd79(34)+acd&
      &79(33)
      acd79(33)=acd79(16)*acd79(33)
      acd79(34)=acd79(25)*acd79(3)
      acd79(35)=acd79(26)*acd79(5)
      acd79(36)=acd79(27)*acd79(7)
      acd79(37)=acd79(28)*acd79(9)
      acd79(38)=-acd79(29)*acd79(11)
      acd79(39)=-acd79(30)*acd79(13)
      acd79(40)=-acd79(31)*acd79(15)
      acd79(34)=acd79(40)+acd79(39)+acd79(38)+acd79(37)+acd79(36)+acd79(35)+acd&
      &79(34)
      acd79(34)=acd79(24)*acd79(34)
      acd79(32)=acd79(34)+acd79(33)+acd79(32)
      brack=2.0_ki*acd79(32)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd79h0_qp
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
      qshift = k3+k4
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
end module     p2_gg_httbar_d79h0l1d_qp
