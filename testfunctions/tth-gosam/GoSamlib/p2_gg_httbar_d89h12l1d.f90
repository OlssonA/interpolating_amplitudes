module     p2_gg_httbar_d89h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d89h12l1d.f90
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
      use p2_gg_httbar_abbrevd89h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(63) :: acd89
      complex(ki) :: brack
      acd89(1)=dotproduct(k2,qshift)
      acd89(2)=abb89(23)
      acd89(3)=dotproduct(qshift,qshift)
      acd89(4)=abb89(25)
      acd89(5)=dotproduct(qshift,spvak2l3)
      acd89(6)=abb89(18)
      acd89(7)=dotproduct(qshift,spvak2l4)
      acd89(8)=abb89(28)
      acd89(9)=dotproduct(qshift,spvak2l5)
      acd89(10)=abb89(14)
      acd89(11)=dotproduct(qshift,spval3k2)
      acd89(12)=abb89(54)
      acd89(13)=dotproduct(qshift,spvak1e2)
      acd89(14)=abb89(27)
      acd89(15)=dotproduct(qshift,spvae2k1)
      acd89(16)=abb89(32)
      acd89(17)=dotproduct(qshift,spvak2e1)
      acd89(18)=dotproduct(qshift,spvae2l3)
      acd89(19)=dotproduct(qshift,spvae1e2)
      acd89(20)=abb89(30)
      acd89(21)=dotproduct(qshift,spvae2l5)
      acd89(22)=abb89(24)
      acd89(23)=abb89(19)
      acd89(24)=abb89(13)
      acd89(25)=abb89(17)
      acd89(26)=abb89(16)
      acd89(27)=abb89(12)
      acd89(28)=dotproduct(qshift,spvae2l4)
      acd89(29)=abb89(26)
      acd89(30)=abb89(31)
      acd89(31)=abb89(46)
      acd89(32)=dotproduct(qshift,spvak2e2)
      acd89(33)=dotproduct(qshift,spvae1l4)
      acd89(34)=dotproduct(qshift,spvae2e1)
      acd89(35)=abb89(9)
      acd89(36)=abb89(11)
      acd89(37)=abb89(10)
      acd89(38)=dotproduct(qshift,spval3e2)
      acd89(39)=abb89(36)
      acd89(40)=abb89(21)
      acd89(41)=abb89(33)
      acd89(42)=abb89(35)
      acd89(43)=abb89(8)
      acd89(44)=abb89(20)
      acd89(45)=abb89(38)
      acd89(46)=abb89(15)
      acd89(47)=-acd89(20)*acd89(18)
      acd89(48)=acd89(22)*acd89(21)
      acd89(47)=acd89(23)+acd89(48)+acd89(47)
      acd89(47)=acd89(47)*acd89(17)
      acd89(48)=acd89(25)*acd89(18)
      acd89(49)=acd89(27)*acd89(21)
      acd89(47)=-acd89(30)+acd89(49)+acd89(48)+acd89(47)
      acd89(47)=acd89(19)*acd89(47)
      acd89(48)=acd89(35)*acd89(32)
      acd89(49)=-acd89(39)*acd89(38)
      acd89(48)=acd89(40)+acd89(49)+acd89(48)
      acd89(48)=acd89(48)*acd89(33)
      acd89(49)=acd89(36)*acd89(32)
      acd89(50)=acd89(42)*acd89(38)
      acd89(48)=-acd89(43)+acd89(50)+acd89(49)+acd89(48)
      acd89(48)=acd89(34)*acd89(48)
      acd89(49)=acd89(29)*acd89(19)
      acd89(49)=-acd89(45)+acd89(49)
      acd89(49)=acd89(28)*acd89(49)
      acd89(50)=-acd89(2)*acd89(1)
      acd89(51)=acd89(4)*acd89(3)
      acd89(52)=-acd89(6)*acd89(5)
      acd89(53)=acd89(8)*acd89(7)
      acd89(54)=-acd89(10)*acd89(9)
      acd89(55)=acd89(12)*acd89(11)
      acd89(56)=-acd89(14)*acd89(13)
      acd89(57)=-acd89(16)*acd89(15)
      acd89(58)=-acd89(24)*acd89(17)
      acd89(59)=-acd89(26)*acd89(18)
      acd89(60)=-acd89(31)*acd89(21)
      acd89(61)=-acd89(37)*acd89(32)
      acd89(62)=-acd89(41)*acd89(33)
      acd89(63)=-acd89(44)*acd89(38)
      brack=acd89(46)+acd89(47)+acd89(48)+acd89(49)+acd89(50)+acd89(51)+acd89(5&
      &2)+acd89(53)+acd89(54)+acd89(55)+acd89(56)+acd89(57)+acd89(58)+acd89(59)&
      &+acd89(60)+acd89(61)+acd89(62)+acd89(63)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd89h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(73) :: acd89
      complex(ki) :: brack
      acd89(1)=k2(iv1)
      acd89(2)=abb89(23)
      acd89(3)=qshift(iv1)
      acd89(4)=abb89(25)
      acd89(5)=spvak2l3(iv1)
      acd89(6)=abb89(18)
      acd89(7)=spvak2l4(iv1)
      acd89(8)=abb89(28)
      acd89(9)=spvak2l5(iv1)
      acd89(10)=abb89(14)
      acd89(11)=spval3k2(iv1)
      acd89(12)=abb89(54)
      acd89(13)=spvak1e2(iv1)
      acd89(14)=abb89(27)
      acd89(15)=spvae2k1(iv1)
      acd89(16)=abb89(32)
      acd89(17)=spvak2e1(iv1)
      acd89(18)=dotproduct(qshift,spvae2l3)
      acd89(19)=dotproduct(qshift,spvae1e2)
      acd89(20)=abb89(30)
      acd89(21)=dotproduct(qshift,spvae2l5)
      acd89(22)=abb89(24)
      acd89(23)=abb89(19)
      acd89(24)=abb89(13)
      acd89(25)=spvae2l3(iv1)
      acd89(26)=dotproduct(qshift,spvak2e1)
      acd89(27)=abb89(17)
      acd89(28)=abb89(16)
      acd89(29)=spvae1e2(iv1)
      acd89(30)=abb89(12)
      acd89(31)=dotproduct(qshift,spvae2l4)
      acd89(32)=abb89(26)
      acd89(33)=abb89(31)
      acd89(34)=spvae2l5(iv1)
      acd89(35)=abb89(46)
      acd89(36)=spvak2e2(iv1)
      acd89(37)=dotproduct(qshift,spvae1l4)
      acd89(38)=dotproduct(qshift,spvae2e1)
      acd89(39)=abb89(9)
      acd89(40)=abb89(11)
      acd89(41)=abb89(10)
      acd89(42)=spvae1l4(iv1)
      acd89(43)=dotproduct(qshift,spvak2e2)
      acd89(44)=dotproduct(qshift,spval3e2)
      acd89(45)=abb89(36)
      acd89(46)=abb89(21)
      acd89(47)=abb89(33)
      acd89(48)=spvae2e1(iv1)
      acd89(49)=abb89(35)
      acd89(50)=abb89(8)
      acd89(51)=spval3e2(iv1)
      acd89(52)=abb89(20)
      acd89(53)=spvae2l4(iv1)
      acd89(54)=abb89(38)
      acd89(55)=acd89(21)*acd89(22)
      acd89(56)=acd89(18)*acd89(20)
      acd89(55)=acd89(23)+acd89(55)-acd89(56)
      acd89(56)=-acd89(17)*acd89(55)
      acd89(57)=-acd89(22)*acd89(34)
      acd89(58)=acd89(20)*acd89(25)
      acd89(57)=acd89(57)+acd89(58)
      acd89(57)=acd89(26)*acd89(57)
      acd89(58)=-acd89(32)*acd89(53)
      acd89(59)=-acd89(34)*acd89(30)
      acd89(60)=-acd89(25)*acd89(27)
      acd89(56)=acd89(57)+acd89(56)+acd89(60)+acd89(58)+acd89(59)
      acd89(56)=acd89(19)*acd89(56)
      acd89(57)=acd89(44)*acd89(45)
      acd89(58)=acd89(39)*acd89(43)
      acd89(57)=-acd89(46)+acd89(57)-acd89(58)
      acd89(58)=acd89(42)*acd89(57)
      acd89(59)=acd89(45)*acd89(51)
      acd89(60)=-acd89(36)*acd89(39)
      acd89(59)=acd89(59)+acd89(60)
      acd89(59)=acd89(37)*acd89(59)
      acd89(60)=-acd89(51)*acd89(49)
      acd89(61)=-acd89(36)*acd89(40)
      acd89(58)=acd89(59)+acd89(58)+acd89(60)+acd89(61)
      acd89(58)=acd89(38)*acd89(58)
      acd89(55)=-acd89(26)*acd89(55)
      acd89(59)=-acd89(32)*acd89(31)
      acd89(60)=-acd89(21)*acd89(30)
      acd89(61)=-acd89(18)*acd89(27)
      acd89(55)=acd89(55)+acd89(61)+acd89(60)+acd89(33)+acd89(59)
      acd89(55)=acd89(29)*acd89(55)
      acd89(57)=acd89(37)*acd89(57)
      acd89(59)=-acd89(44)*acd89(49)
      acd89(60)=-acd89(43)*acd89(40)
      acd89(57)=acd89(57)+acd89(60)+acd89(50)+acd89(59)
      acd89(57)=acd89(48)*acd89(57)
      acd89(59)=acd89(15)*acd89(16)
      acd89(60)=acd89(13)*acd89(14)
      acd89(61)=-acd89(11)*acd89(12)
      acd89(62)=acd89(9)*acd89(10)
      acd89(63)=-acd89(7)*acd89(8)
      acd89(64)=acd89(5)*acd89(6)
      acd89(65)=acd89(3)*acd89(4)
      acd89(66)=acd89(1)*acd89(2)
      acd89(67)=acd89(53)*acd89(54)
      acd89(68)=acd89(51)*acd89(52)
      acd89(69)=acd89(36)*acd89(41)
      acd89(70)=acd89(34)*acd89(35)
      acd89(71)=acd89(25)*acd89(28)
      acd89(72)=acd89(42)*acd89(47)
      acd89(73)=acd89(17)*acd89(24)
      brack=acd89(55)+acd89(56)+acd89(57)+acd89(58)+acd89(59)+acd89(60)+acd89(6&
      &1)+acd89(62)+acd89(63)+acd89(64)-2.0_ki*acd89(65)+acd89(66)+acd89(67)+ac&
      &d89(68)+acd89(69)+acd89(70)+acd89(71)+acd89(72)+acd89(73)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd89h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(53) :: acd89
      complex(ki) :: brack
      acd89(1)=d(iv1,iv2)
      acd89(2)=abb89(25)
      acd89(3)=spvak2e1(iv1)
      acd89(4)=spvae2l3(iv2)
      acd89(5)=dotproduct(qshift,spvae1e2)
      acd89(6)=abb89(30)
      acd89(7)=spvae1e2(iv2)
      acd89(8)=dotproduct(qshift,spvae2l3)
      acd89(9)=dotproduct(qshift,spvae2l5)
      acd89(10)=abb89(24)
      acd89(11)=abb89(19)
      acd89(12)=spvae2l5(iv2)
      acd89(13)=spvak2e1(iv2)
      acd89(14)=spvae2l3(iv1)
      acd89(15)=spvae1e2(iv1)
      acd89(16)=spvae2l5(iv1)
      acd89(17)=dotproduct(qshift,spvak2e1)
      acd89(18)=abb89(17)
      acd89(19)=abb89(12)
      acd89(20)=spvae2l4(iv2)
      acd89(21)=abb89(26)
      acd89(22)=spvae2l4(iv1)
      acd89(23)=spvak2e2(iv1)
      acd89(24)=spvae1l4(iv2)
      acd89(25)=dotproduct(qshift,spvae2e1)
      acd89(26)=abb89(9)
      acd89(27)=spvae2e1(iv2)
      acd89(28)=dotproduct(qshift,spvae1l4)
      acd89(29)=abb89(11)
      acd89(30)=spvak2e2(iv2)
      acd89(31)=spvae1l4(iv1)
      acd89(32)=spvae2e1(iv1)
      acd89(33)=dotproduct(qshift,spvak2e2)
      acd89(34)=dotproduct(qshift,spval3e2)
      acd89(35)=abb89(36)
      acd89(36)=abb89(21)
      acd89(37)=spval3e2(iv2)
      acd89(38)=spval3e2(iv1)
      acd89(39)=abb89(35)
      acd89(40)=-acd89(4)*acd89(6)
      acd89(41)=acd89(12)*acd89(10)
      acd89(40)=acd89(40)+acd89(41)
      acd89(41)=acd89(5)*acd89(3)
      acd89(42)=acd89(17)*acd89(15)
      acd89(41)=acd89(41)+acd89(42)
      acd89(40)=acd89(41)*acd89(40)
      acd89(41)=-acd89(14)*acd89(6)
      acd89(42)=acd89(16)*acd89(10)
      acd89(41)=acd89(42)+acd89(41)
      acd89(42)=acd89(5)*acd89(13)
      acd89(43)=acd89(17)*acd89(7)
      acd89(42)=acd89(42)+acd89(43)
      acd89(41)=acd89(42)*acd89(41)
      acd89(42)=-acd89(8)*acd89(6)
      acd89(43)=acd89(9)*acd89(10)
      acd89(42)=acd89(11)+acd89(43)+acd89(42)
      acd89(43)=acd89(3)*acd89(7)
      acd89(44)=acd89(13)*acd89(15)
      acd89(43)=acd89(43)+acd89(44)
      acd89(42)=acd89(43)*acd89(42)
      acd89(43)=acd89(33)*acd89(26)
      acd89(44)=-acd89(34)*acd89(35)
      acd89(43)=acd89(36)+acd89(44)+acd89(43)
      acd89(44)=acd89(24)*acd89(32)
      acd89(45)=acd89(31)*acd89(27)
      acd89(44)=acd89(44)+acd89(45)
      acd89(43)=acd89(44)*acd89(43)
      acd89(44)=acd89(20)*acd89(15)
      acd89(45)=acd89(22)*acd89(7)
      acd89(44)=acd89(45)+acd89(44)
      acd89(44)=acd89(21)*acd89(44)
      acd89(45)=acd89(25)*acd89(24)
      acd89(46)=acd89(26)*acd89(45)
      acd89(47)=acd89(28)*acd89(26)
      acd89(48)=acd89(27)*acd89(47)
      acd89(46)=acd89(46)+acd89(48)
      acd89(46)=acd89(23)*acd89(46)
      acd89(48)=acd89(25)*acd89(31)
      acd89(49)=acd89(26)*acd89(48)
      acd89(47)=acd89(32)*acd89(47)
      acd89(47)=acd89(49)+acd89(47)
      acd89(47)=acd89(30)*acd89(47)
      acd89(48)=-acd89(35)*acd89(48)
      acd89(49)=acd89(28)*acd89(35)
      acd89(50)=-acd89(32)*acd89(49)
      acd89(48)=acd89(48)+acd89(50)
      acd89(48)=acd89(37)*acd89(48)
      acd89(45)=-acd89(35)*acd89(45)
      acd89(49)=-acd89(27)*acd89(49)
      acd89(45)=acd89(45)+acd89(49)
      acd89(45)=acd89(38)*acd89(45)
      acd89(49)=acd89(4)*acd89(15)
      acd89(50)=acd89(14)*acd89(7)
      acd89(49)=acd89(49)+acd89(50)
      acd89(49)=acd89(18)*acd89(49)
      acd89(50)=acd89(12)*acd89(15)
      acd89(51)=acd89(16)*acd89(7)
      acd89(50)=acd89(50)+acd89(51)
      acd89(50)=acd89(19)*acd89(50)
      acd89(51)=acd89(23)*acd89(27)
      acd89(52)=acd89(30)*acd89(32)
      acd89(51)=acd89(51)+acd89(52)
      acd89(51)=acd89(29)*acd89(51)
      acd89(52)=acd89(37)*acd89(32)
      acd89(53)=acd89(38)*acd89(27)
      acd89(52)=acd89(52)+acd89(53)
      acd89(52)=acd89(39)*acd89(52)
      acd89(53)=acd89(2)*acd89(1)
      brack=acd89(40)+acd89(41)+acd89(42)+acd89(43)+acd89(44)+acd89(45)+acd89(4&
      &6)+acd89(47)+acd89(48)+acd89(49)+acd89(50)+acd89(51)+acd89(52)+2.0_ki*ac&
      &d89(53)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd89h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd89
      complex(ki) :: brack
      acd89(1)=spvak2e1(iv1)
      acd89(2)=spvae2l3(iv2)
      acd89(3)=spvae1e2(iv3)
      acd89(4)=abb89(30)
      acd89(5)=spvae2l3(iv3)
      acd89(6)=spvae1e2(iv2)
      acd89(7)=spvae2l5(iv3)
      acd89(8)=abb89(24)
      acd89(9)=spvae2l5(iv2)
      acd89(10)=spvak2e1(iv2)
      acd89(11)=spvae2l3(iv1)
      acd89(12)=spvae1e2(iv1)
      acd89(13)=spvae2l5(iv1)
      acd89(14)=spvak2e1(iv3)
      acd89(15)=spvak2e2(iv1)
      acd89(16)=spvae1l4(iv2)
      acd89(17)=spvae2e1(iv3)
      acd89(18)=abb89(9)
      acd89(19)=spvae1l4(iv3)
      acd89(20)=spvae2e1(iv2)
      acd89(21)=spvak2e2(iv2)
      acd89(22)=spvae1l4(iv1)
      acd89(23)=spvae2e1(iv1)
      acd89(24)=spvak2e2(iv3)
      acd89(25)=spval3e2(iv3)
      acd89(26)=abb89(36)
      acd89(27)=spval3e2(iv2)
      acd89(28)=spval3e2(iv1)
      acd89(29)=acd89(3)*acd89(1)
      acd89(30)=acd89(14)*acd89(12)
      acd89(29)=acd89(29)+acd89(30)
      acd89(30)=acd89(2)*acd89(29)
      acd89(31)=acd89(6)*acd89(1)
      acd89(32)=acd89(12)*acd89(10)
      acd89(31)=acd89(31)+acd89(32)
      acd89(32)=acd89(5)*acd89(31)
      acd89(33)=acd89(10)*acd89(3)
      acd89(34)=acd89(14)*acd89(6)
      acd89(33)=acd89(33)+acd89(34)
      acd89(34)=acd89(11)*acd89(33)
      acd89(30)=acd89(34)+acd89(30)+acd89(32)
      acd89(30)=acd89(4)*acd89(30)
      acd89(31)=-acd89(7)*acd89(31)
      acd89(29)=-acd89(9)*acd89(29)
      acd89(32)=-acd89(13)*acd89(33)
      acd89(29)=acd89(32)+acd89(29)+acd89(31)
      acd89(29)=acd89(8)*acd89(29)
      acd89(31)=acd89(17)*acd89(16)
      acd89(32)=acd89(20)*acd89(19)
      acd89(31)=acd89(31)+acd89(32)
      acd89(32)=-acd89(15)*acd89(31)
      acd89(33)=acd89(22)*acd89(17)
      acd89(34)=acd89(23)*acd89(19)
      acd89(33)=acd89(33)+acd89(34)
      acd89(34)=-acd89(21)*acd89(33)
      acd89(35)=acd89(22)*acd89(20)
      acd89(36)=acd89(23)*acd89(16)
      acd89(35)=acd89(35)+acd89(36)
      acd89(36)=-acd89(24)*acd89(35)
      acd89(32)=acd89(36)+acd89(34)+acd89(32)
      acd89(32)=acd89(18)*acd89(32)
      acd89(34)=acd89(25)*acd89(35)
      acd89(33)=acd89(27)*acd89(33)
      acd89(31)=acd89(28)*acd89(31)
      acd89(31)=acd89(31)+acd89(33)+acd89(34)
      acd89(31)=acd89(26)*acd89(31)
      brack=acd89(29)+acd89(30)+acd89(31)+acd89(32)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd89h12
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
end module     p2_gg_httbar_d89h12l1d
