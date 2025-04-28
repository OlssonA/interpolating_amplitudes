module     p2_gg_httbar_d92h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d92h0l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd92h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc92(69)
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: QspQ
      complex(ki) :: Qspl4
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspe1
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      QspQ = dotproduct(Q,Q)
      Qspl4 = dotproduct(Q,l4)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspe1 = dotproduct(Q,e1)
      acc92(1)=abb92(8)
      acc92(2)=abb92(9)
      acc92(3)=abb92(10)
      acc92(4)=abb92(11)
      acc92(5)=abb92(12)
      acc92(6)=abb92(13)
      acc92(7)=abb92(14)
      acc92(8)=abb92(15)
      acc92(9)=abb92(16)
      acc92(10)=abb92(17)
      acc92(11)=abb92(18)
      acc92(12)=abb92(19)
      acc92(13)=abb92(20)
      acc92(14)=abb92(21)
      acc92(15)=abb92(22)
      acc92(16)=abb92(23)
      acc92(17)=abb92(24)
      acc92(18)=abb92(25)
      acc92(19)=abb92(26)
      acc92(20)=abb92(27)
      acc92(21)=abb92(28)
      acc92(22)=abb92(29)
      acc92(23)=abb92(30)
      acc92(24)=abb92(31)
      acc92(25)=abb92(32)
      acc92(26)=abb92(33)
      acc92(27)=abb92(34)
      acc92(28)=abb92(35)
      acc92(29)=abb92(36)
      acc92(30)=abb92(37)
      acc92(31)=abb92(39)
      acc92(32)=abb92(40)
      acc92(33)=abb92(42)
      acc92(34)=abb92(43)
      acc92(35)=abb92(44)
      acc92(36)=abb92(45)
      acc92(37)=abb92(46)
      acc92(38)=abb92(48)
      acc92(39)=abb92(49)
      acc92(40)=abb92(51)
      acc92(41)=abb92(53)
      acc92(42)=abb92(55)
      acc92(43)=abb92(57)
      acc92(44)=abb92(58)
      acc92(45)=abb92(63)
      acc92(46)=abb92(68)
      acc92(47)=abb92(69)
      acc92(48)=abb92(70)
      acc92(49)=abb92(73)
      acc92(50)=acc92(16)*Qspval4k2
      acc92(51)=acc92(17)*Qspvae2k2
      acc92(52)=acc92(22)*Qspval3l4
      acc92(53)=acc92(25)*Qspval3k2
      acc92(54)=acc92(27)*Qspvae2k1
      acc92(55)=acc92(30)*Qspval5l4
      acc92(56)=acc92(31)*Qspval5k2
      acc92(57)=acc92(36)*QspQ
      acc92(58)=acc92(38)*Qspl4
      acc92(59)=acc92(45)*Qspvae2l3
      acc92(50)=acc92(59)+acc92(58)+acc92(57)+acc92(56)+acc92(55)+acc92(54)+acc&
      &92(53)+acc92(52)+acc92(21)+acc92(51)+acc92(50)
      acc92(50)=Qspvae1e2*acc92(50)
      acc92(51)=acc92(7)*Qspval4k2
      acc92(52)=acc92(8)*Qspvak1e2
      acc92(53)=acc92(18)*Qspk2
      acc92(54)=acc92(28)*Qspvak2l3
      acc92(55)=acc92(29)*Qspval4e2
      acc92(56)=acc92(35)*QspQ
      acc92(57)=acc92(43)*Qspval4l3
      acc92(58)=acc92(44)*Qspval5e2
      acc92(59)=acc92(49)*Qspval3e2
      acc92(51)=acc92(59)+acc92(58)+acc92(57)+acc92(56)+acc92(34)+acc92(55)+acc&
      &92(54)+acc92(53)+acc92(52)+acc92(51)
      acc92(51)=Qspvae2e1*acc92(51)
      acc92(52)=acc92(1)*Qspval4e2
      acc92(53)=acc92(5)*Qspval3e2
      acc92(54)=acc92(24)*Qspval5e2
      acc92(52)=acc92(54)+acc92(53)+acc92(4)+acc92(52)
      acc92(52)=acc92(52)*Qspvae2k2
      acc92(53)=acc92(46)*Qspvae2l3
      acc92(53)=acc92(53)+acc92(10)
      acc92(53)=acc92(53)*Qspval4e2
      acc92(54)=acc92(2)*Qspval5e2
      acc92(55)=acc92(3)*Qspvae2l3
      acc92(56)=acc92(48)*Qspval3e2
      acc92(52)=acc92(56)+acc92(14)+acc92(55)+acc92(54)+acc92(52)+acc92(53)
      acc92(52)=Qspe1*acc92(52)
      acc92(53)=acc92(9)*Qspvak1e2
      acc92(54)=acc92(11)*Qspvae2k2
      acc92(55)=acc92(12)*Qspval4k2
      acc92(56)=acc92(13)*Qspval3k2
      acc92(57)=acc92(15)*QspQ
      acc92(58)=acc92(19)*Qspval3l4
      acc92(59)=acc92(20)*Qspval4e2
      acc92(60)=acc92(23)*Qspval5l4
      acc92(61)=acc92(26)*Qspvak2l3
      acc92(62)=acc92(32)*Qspl4
      acc92(63)=acc92(33)*Qspvae2k1
      acc92(64)=acc92(37)*Qspval5k2
      acc92(65)=acc92(39)*Qspval4l3
      acc92(66)=acc92(40)*Qspk2
      acc92(67)=acc92(41)*Qspvae2l3
      acc92(68)=acc92(42)*Qspval5e2
      acc92(69)=acc92(47)*Qspval3e2
      brack=acc92(6)+acc92(50)+acc92(51)+acc92(52)+acc92(53)+acc92(54)+acc92(55&
      &)+acc92(56)+acc92(57)+acc92(58)+acc92(59)+acc92(60)+acc92(61)+acc92(62)+&
      &acc92(63)+acc92(64)+acc92(65)+acc92(66)+acc92(67)+acc92(68)+acc92(69)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d92h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd92h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d92
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d92 = 0.0_ki
      d92 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d92, ki), aimag(d92), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d92h0l1_qp
