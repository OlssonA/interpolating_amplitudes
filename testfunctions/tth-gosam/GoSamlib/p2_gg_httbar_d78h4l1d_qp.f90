module     p2_gg_httbar_d78h4l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d78h4l1d_qp.f90
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
      use p2_gg_httbar_abbrevd78h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(47) :: acd78
      complex(ki) :: brack
      acd78(1)=dotproduct(qshift,qshift)
      acd78(2)=dotproduct(qshift,spvak2e1)
      acd78(3)=abb78(15)
      acd78(4)=dotproduct(qshift,spvae1l4)
      acd78(5)=abb78(40)
      acd78(6)=dotproduct(qshift,spval5e1)
      acd78(7)=abb78(27)
      acd78(8)=dotproduct(qshift,spvae1l5)
      acd78(9)=abb78(41)
      acd78(10)=dotproduct(qshift,spvae1e2)
      acd78(11)=abb78(36)
      acd78(12)=dotproduct(qshift,spvae2e1)
      acd78(13)=abb78(29)
      acd78(14)=abb78(19)
      acd78(15)=abb78(14)
      acd78(16)=abb78(24)
      acd78(17)=abb78(22)
      acd78(18)=dotproduct(qshift,spvae1k2)
      acd78(19)=abb78(21)
      acd78(20)=dotproduct(qshift,spvae1l3)
      acd78(21)=abb78(18)
      acd78(22)=abb78(13)
      acd78(23)=abb78(31)
      acd78(24)=abb78(20)
      acd78(25)=dotproduct(qshift,spval3e1)
      acd78(26)=abb78(11)
      acd78(27)=abb78(9)
      acd78(28)=abb78(38)
      acd78(29)=abb78(23)
      acd78(30)=abb78(42)
      acd78(31)=abb78(34)
      acd78(32)=abb78(39)
      acd78(33)=abb78(16)
      acd78(34)=abb78(37)
      acd78(35)=abb78(25)
      acd78(36)=abb78(12)
      acd78(37)=abb78(17)
      acd78(38)=abb78(10)
      acd78(39)=-acd78(3)*acd78(1)
      acd78(40)=-acd78(15)*acd78(4)
      acd78(41)=acd78(16)*acd78(8)
      acd78(42)=acd78(17)*acd78(10)
      acd78(43)=acd78(19)*acd78(18)
      acd78(44)=acd78(21)*acd78(20)
      acd78(39)=-acd78(22)+acd78(44)+acd78(43)+acd78(42)+acd78(41)+acd78(40)+ac&
      &d78(39)
      acd78(39)=acd78(2)*acd78(39)
      acd78(40)=acd78(5)*acd78(4)
      acd78(41)=-acd78(7)*acd78(6)
      acd78(42)=-acd78(9)*acd78(8)
      acd78(43)=-acd78(11)*acd78(10)
      acd78(44)=-acd78(13)*acd78(12)
      acd78(40)=acd78(14)+acd78(44)+acd78(43)+acd78(42)+acd78(41)+acd78(40)
      acd78(40)=acd78(1)*acd78(40)
      acd78(41)=acd78(26)*acd78(4)
      acd78(42)=-acd78(30)*acd78(8)
      acd78(43)=-acd78(32)*acd78(10)
      acd78(41)=-acd78(37)+acd78(43)+acd78(42)+acd78(41)
      acd78(41)=acd78(25)*acd78(41)
      acd78(42)=acd78(23)*acd78(6)
      acd78(43)=acd78(24)*acd78(12)
      acd78(42)=-acd78(27)+acd78(43)+acd78(42)
      acd78(42)=acd78(4)*acd78(42)
      acd78(43)=acd78(28)*acd78(6)
      acd78(44)=acd78(34)*acd78(12)
      acd78(43)=-acd78(36)+acd78(44)+acd78(43)
      acd78(43)=acd78(20)*acd78(43)
      acd78(44)=-acd78(29)*acd78(6)
      acd78(45)=-acd78(31)*acd78(8)
      acd78(46)=-acd78(33)*acd78(10)
      acd78(47)=-acd78(35)*acd78(12)
      brack=acd78(38)+acd78(39)+acd78(40)+acd78(41)+acd78(42)+acd78(43)+acd78(4&
      &4)+acd78(45)+acd78(46)+acd78(47)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd78h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(59) :: acd78
      complex(ki) :: brack
      acd78(1)=qshift(iv1)
      acd78(2)=dotproduct(qshift,spvak2e1)
      acd78(3)=abb78(15)
      acd78(4)=dotproduct(qshift,spvae1l4)
      acd78(5)=abb78(40)
      acd78(6)=dotproduct(qshift,spval5e1)
      acd78(7)=abb78(27)
      acd78(8)=dotproduct(qshift,spvae1l5)
      acd78(9)=abb78(41)
      acd78(10)=dotproduct(qshift,spvae1e2)
      acd78(11)=abb78(36)
      acd78(12)=dotproduct(qshift,spvae2e1)
      acd78(13)=abb78(29)
      acd78(14)=abb78(19)
      acd78(15)=spvak2e1(iv1)
      acd78(16)=dotproduct(qshift,qshift)
      acd78(17)=abb78(14)
      acd78(18)=abb78(24)
      acd78(19)=abb78(22)
      acd78(20)=dotproduct(qshift,spvae1k2)
      acd78(21)=abb78(21)
      acd78(22)=dotproduct(qshift,spvae1l3)
      acd78(23)=abb78(18)
      acd78(24)=abb78(13)
      acd78(25)=spvae1l4(iv1)
      acd78(26)=abb78(31)
      acd78(27)=abb78(20)
      acd78(28)=dotproduct(qshift,spval3e1)
      acd78(29)=abb78(11)
      acd78(30)=abb78(9)
      acd78(31)=spval5e1(iv1)
      acd78(32)=abb78(38)
      acd78(33)=abb78(23)
      acd78(34)=spvae1l5(iv1)
      acd78(35)=abb78(42)
      acd78(36)=abb78(34)
      acd78(37)=spvae1e2(iv1)
      acd78(38)=abb78(39)
      acd78(39)=abb78(16)
      acd78(40)=spvae2e1(iv1)
      acd78(41)=abb78(37)
      acd78(42)=abb78(25)
      acd78(43)=spvae1k2(iv1)
      acd78(44)=spvae1l3(iv1)
      acd78(45)=abb78(12)
      acd78(46)=spval3e1(iv1)
      acd78(47)=abb78(17)
      acd78(48)=-acd78(21)*acd78(20)
      acd78(49)=-acd78(22)*acd78(23)
      acd78(50)=-acd78(10)*acd78(19)
      acd78(51)=-acd78(8)*acd78(18)
      acd78(52)=acd78(4)*acd78(17)
      acd78(53)=acd78(16)*acd78(3)
      acd78(48)=acd78(53)+acd78(52)+acd78(51)+acd78(50)+acd78(49)+acd78(24)+acd&
      &78(48)
      acd78(48)=acd78(15)*acd78(48)
      acd78(49)=acd78(12)*acd78(13)
      acd78(50)=acd78(10)*acd78(11)
      acd78(51)=acd78(8)*acd78(9)
      acd78(52)=acd78(6)*acd78(7)
      acd78(53)=-acd78(4)*acd78(5)
      acd78(54)=acd78(2)*acd78(3)
      acd78(49)=acd78(54)+acd78(53)+acd78(52)+acd78(51)+acd78(50)-acd78(14)+acd&
      &78(49)
      acd78(49)=acd78(1)*acd78(49)
      acd78(50)=acd78(40)*acd78(13)
      acd78(51)=acd78(37)*acd78(11)
      acd78(52)=acd78(34)*acd78(9)
      acd78(53)=acd78(31)*acd78(7)
      acd78(54)=-acd78(25)*acd78(5)
      acd78(50)=acd78(54)+acd78(53)+acd78(52)+acd78(50)+acd78(51)
      acd78(50)=acd78(16)*acd78(50)
      acd78(51)=-acd78(21)*acd78(43)
      acd78(52)=-acd78(44)*acd78(23)
      acd78(53)=-acd78(37)*acd78(19)
      acd78(54)=-acd78(34)*acd78(18)
      acd78(55)=acd78(25)*acd78(17)
      acd78(51)=acd78(55)+acd78(54)+acd78(53)+acd78(51)+acd78(52)
      acd78(51)=acd78(2)*acd78(51)
      acd78(52)=-acd78(46)*acd78(29)
      acd78(53)=-acd78(40)*acd78(27)
      acd78(54)=-acd78(31)*acd78(26)
      acd78(52)=acd78(54)+acd78(52)+acd78(53)
      acd78(52)=acd78(4)*acd78(52)
      acd78(53)=-acd78(28)*acd78(29)
      acd78(54)=-acd78(12)*acd78(27)
      acd78(55)=-acd78(6)*acd78(26)
      acd78(53)=acd78(55)+acd78(54)+acd78(30)+acd78(53)
      acd78(53)=acd78(25)*acd78(53)
      acd78(54)=acd78(10)*acd78(38)
      acd78(55)=acd78(8)*acd78(35)
      acd78(54)=acd78(55)+acd78(47)+acd78(54)
      acd78(54)=acd78(46)*acd78(54)
      acd78(55)=-acd78(12)*acd78(41)
      acd78(56)=-acd78(6)*acd78(32)
      acd78(55)=acd78(56)+acd78(45)+acd78(55)
      acd78(55)=acd78(44)*acd78(55)
      acd78(56)=-acd78(22)*acd78(41)
      acd78(56)=acd78(42)+acd78(56)
      acd78(56)=acd78(40)*acd78(56)
      acd78(57)=acd78(28)*acd78(38)
      acd78(57)=acd78(39)+acd78(57)
      acd78(57)=acd78(37)*acd78(57)
      acd78(58)=acd78(28)*acd78(35)
      acd78(58)=acd78(36)+acd78(58)
      acd78(58)=acd78(34)*acd78(58)
      acd78(59)=-acd78(22)*acd78(32)
      acd78(59)=acd78(33)+acd78(59)
      acd78(59)=acd78(31)*acd78(59)
      brack=acd78(48)+2.0_ki*acd78(49)+acd78(50)+acd78(51)+acd78(52)+acd78(53)+&
      &acd78(54)+acd78(55)+acd78(56)+acd78(57)+acd78(58)+acd78(59)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd78h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(56) :: acd78
      complex(ki) :: brack
      acd78(1)=d(iv1,iv2)
      acd78(2)=dotproduct(qshift,spvak2e1)
      acd78(3)=abb78(15)
      acd78(4)=dotproduct(qshift,spvae1l4)
      acd78(5)=abb78(40)
      acd78(6)=dotproduct(qshift,spval5e1)
      acd78(7)=abb78(27)
      acd78(8)=dotproduct(qshift,spvae1l5)
      acd78(9)=abb78(41)
      acd78(10)=dotproduct(qshift,spvae1e2)
      acd78(11)=abb78(36)
      acd78(12)=dotproduct(qshift,spvae2e1)
      acd78(13)=abb78(29)
      acd78(14)=abb78(19)
      acd78(15)=qshift(iv1)
      acd78(16)=spvak2e1(iv2)
      acd78(17)=spvae1l4(iv2)
      acd78(18)=spval5e1(iv2)
      acd78(19)=spvae1l5(iv2)
      acd78(20)=spvae1e2(iv2)
      acd78(21)=spvae2e1(iv2)
      acd78(22)=qshift(iv2)
      acd78(23)=spvak2e1(iv1)
      acd78(24)=spvae1l4(iv1)
      acd78(25)=spval5e1(iv1)
      acd78(26)=spvae1l5(iv1)
      acd78(27)=spvae1e2(iv1)
      acd78(28)=spvae2e1(iv1)
      acd78(29)=abb78(14)
      acd78(30)=abb78(24)
      acd78(31)=abb78(22)
      acd78(32)=spvae1k2(iv2)
      acd78(33)=abb78(21)
      acd78(34)=spvae1l3(iv2)
      acd78(35)=abb78(18)
      acd78(36)=spvae1k2(iv1)
      acd78(37)=spvae1l3(iv1)
      acd78(38)=abb78(31)
      acd78(39)=abb78(20)
      acd78(40)=spval3e1(iv2)
      acd78(41)=abb78(11)
      acd78(42)=spval3e1(iv1)
      acd78(43)=abb78(38)
      acd78(44)=abb78(42)
      acd78(45)=abb78(39)
      acd78(46)=abb78(37)
      acd78(47)=-acd78(13)*acd78(28)
      acd78(48)=-acd78(11)*acd78(27)
      acd78(49)=-acd78(9)*acd78(26)
      acd78(50)=-acd78(7)*acd78(25)
      acd78(51)=acd78(24)*acd78(5)
      acd78(52)=-acd78(23)*acd78(3)
      acd78(47)=acd78(52)+acd78(51)+acd78(50)+acd78(49)+acd78(47)+acd78(48)
      acd78(47)=acd78(22)*acd78(47)
      acd78(48)=-acd78(13)*acd78(21)
      acd78(49)=-acd78(11)*acd78(20)
      acd78(50)=-acd78(9)*acd78(19)
      acd78(51)=-acd78(7)*acd78(18)
      acd78(52)=acd78(17)*acd78(5)
      acd78(53)=-acd78(16)*acd78(3)
      acd78(48)=acd78(53)+acd78(52)+acd78(51)+acd78(50)+acd78(48)+acd78(49)
      acd78(48)=acd78(15)*acd78(48)
      acd78(49)=-acd78(13)*acd78(12)
      acd78(50)=-acd78(11)*acd78(10)
      acd78(51)=-acd78(9)*acd78(8)
      acd78(52)=-acd78(7)*acd78(6)
      acd78(53)=acd78(5)*acd78(4)
      acd78(54)=-acd78(3)*acd78(2)
      acd78(49)=acd78(54)+acd78(53)+acd78(52)+acd78(51)+acd78(50)+acd78(14)+acd&
      &78(49)
      acd78(49)=acd78(1)*acd78(49)
      acd78(47)=acd78(49)+acd78(47)+acd78(48)
      acd78(48)=acd78(33)*acd78(32)
      acd78(49)=acd78(34)*acd78(35)
      acd78(50)=acd78(20)*acd78(31)
      acd78(51)=acd78(19)*acd78(30)
      acd78(52)=-acd78(17)*acd78(29)
      acd78(48)=acd78(52)+acd78(51)+acd78(50)+acd78(48)+acd78(49)
      acd78(48)=acd78(23)*acd78(48)
      acd78(49)=acd78(33)*acd78(36)
      acd78(50)=acd78(37)*acd78(35)
      acd78(51)=acd78(27)*acd78(31)
      acd78(52)=acd78(26)*acd78(30)
      acd78(53)=-acd78(24)*acd78(29)
      acd78(49)=acd78(53)+acd78(52)+acd78(51)+acd78(49)+acd78(50)
      acd78(49)=acd78(16)*acd78(49)
      acd78(50)=acd78(40)*acd78(41)
      acd78(51)=acd78(21)*acd78(39)
      acd78(52)=acd78(18)*acd78(38)
      acd78(50)=acd78(52)+acd78(50)+acd78(51)
      acd78(50)=acd78(24)*acd78(50)
      acd78(51)=acd78(42)*acd78(41)
      acd78(52)=acd78(28)*acd78(39)
      acd78(53)=acd78(25)*acd78(38)
      acd78(51)=acd78(53)+acd78(51)+acd78(52)
      acd78(51)=acd78(17)*acd78(51)
      acd78(52)=-acd78(20)*acd78(45)
      acd78(53)=-acd78(19)*acd78(44)
      acd78(52)=acd78(53)+acd78(52)
      acd78(52)=acd78(42)*acd78(52)
      acd78(53)=-acd78(27)*acd78(45)
      acd78(54)=-acd78(26)*acd78(44)
      acd78(53)=acd78(54)+acd78(53)
      acd78(53)=acd78(40)*acd78(53)
      acd78(54)=acd78(21)*acd78(46)
      acd78(55)=acd78(18)*acd78(43)
      acd78(54)=acd78(55)+acd78(54)
      acd78(54)=acd78(37)*acd78(54)
      acd78(55)=acd78(28)*acd78(46)
      acd78(56)=acd78(25)*acd78(43)
      acd78(55)=acd78(56)+acd78(55)
      acd78(55)=acd78(34)*acd78(55)
      brack=2.0_ki*acd78(47)+acd78(48)+acd78(49)+acd78(50)+acd78(51)+acd78(52)+&
      &acd78(53)+acd78(54)+acd78(55)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd78h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd78
      complex(ki) :: brack
      acd78(1)=d(iv1,iv2)
      acd78(2)=spvak2e1(iv3)
      acd78(3)=abb78(15)
      acd78(4)=spvae1l4(iv3)
      acd78(5)=abb78(40)
      acd78(6)=spval5e1(iv3)
      acd78(7)=abb78(27)
      acd78(8)=spvae1l5(iv3)
      acd78(9)=abb78(41)
      acd78(10)=spvae1e2(iv3)
      acd78(11)=abb78(36)
      acd78(12)=spvae2e1(iv3)
      acd78(13)=abb78(29)
      acd78(14)=d(iv1,iv3)
      acd78(15)=spvak2e1(iv2)
      acd78(16)=spvae1l4(iv2)
      acd78(17)=spval5e1(iv2)
      acd78(18)=spvae1l5(iv2)
      acd78(19)=spvae1e2(iv2)
      acd78(20)=spvae2e1(iv2)
      acd78(21)=d(iv2,iv3)
      acd78(22)=spvak2e1(iv1)
      acd78(23)=spvae1l4(iv1)
      acd78(24)=spval5e1(iv1)
      acd78(25)=spvae1l5(iv1)
      acd78(26)=spvae1e2(iv1)
      acd78(27)=spvae2e1(iv1)
      acd78(28)=acd78(2)*acd78(3)
      acd78(29)=-acd78(4)*acd78(5)
      acd78(30)=acd78(6)*acd78(7)
      acd78(31)=acd78(8)*acd78(9)
      acd78(32)=acd78(10)*acd78(11)
      acd78(33)=acd78(12)*acd78(13)
      acd78(28)=acd78(33)+acd78(32)+acd78(31)+acd78(30)+acd78(28)+acd78(29)
      acd78(28)=acd78(1)*acd78(28)
      acd78(29)=acd78(15)*acd78(3)
      acd78(30)=-acd78(16)*acd78(5)
      acd78(31)=acd78(17)*acd78(7)
      acd78(32)=acd78(18)*acd78(9)
      acd78(33)=acd78(19)*acd78(11)
      acd78(34)=acd78(20)*acd78(13)
      acd78(29)=acd78(34)+acd78(33)+acd78(32)+acd78(31)+acd78(30)+acd78(29)
      acd78(29)=acd78(14)*acd78(29)
      acd78(30)=acd78(22)*acd78(3)
      acd78(31)=-acd78(23)*acd78(5)
      acd78(32)=acd78(24)*acd78(7)
      acd78(33)=acd78(25)*acd78(9)
      acd78(34)=acd78(26)*acd78(11)
      acd78(35)=acd78(27)*acd78(13)
      acd78(30)=acd78(35)+acd78(34)+acd78(33)+acd78(32)+acd78(31)+acd78(30)
      acd78(30)=acd78(21)*acd78(30)
      acd78(28)=acd78(30)+acd78(29)+acd78(28)
      brack=2.0_ki*acd78(28)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd78h4_qp
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
end module     p2_gg_httbar_d78h4l1d_qp
