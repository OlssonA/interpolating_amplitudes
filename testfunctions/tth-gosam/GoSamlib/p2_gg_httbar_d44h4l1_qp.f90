module     p2_gg_httbar_d44h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d44h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd44h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc44(59)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      complex(ki) :: Qspk1
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      Qspk1 = dotproduct(Q,k1)
      acc44(1)=abb44(14)
      acc44(2)=abb44(15)
      acc44(3)=abb44(16)
      acc44(4)=abb44(17)
      acc44(5)=abb44(18)
      acc44(6)=abb44(19)
      acc44(7)=abb44(21)
      acc44(8)=abb44(23)
      acc44(9)=abb44(25)
      acc44(10)=abb44(26)
      acc44(11)=abb44(28)
      acc44(12)=abb44(30)
      acc44(13)=abb44(31)
      acc44(14)=abb44(32)
      acc44(15)=abb44(33)
      acc44(16)=abb44(34)
      acc44(17)=abb44(35)
      acc44(18)=abb44(36)
      acc44(19)=abb44(37)
      acc44(20)=abb44(42)
      acc44(21)=abb44(43)
      acc44(22)=abb44(44)
      acc44(23)=abb44(46)
      acc44(24)=abb44(48)
      acc44(25)=abb44(49)
      acc44(26)=abb44(50)
      acc44(27)=abb44(55)
      acc44(28)=abb44(57)
      acc44(29)=abb44(59)
      acc44(30)=abb44(60)
      acc44(31)=Qspvae2e1*acc44(7)
      acc44(32)=Qspvae1e2*acc44(11)
      acc44(33)=Qspvae1l5*acc44(15)
      acc44(34)=Qspval5e1*acc44(9)
      acc44(35)=Qspvae2l4*acc44(18)
      acc44(36)=Qspval4e2*acc44(10)
      acc44(37)=Qspvae1l4*acc44(13)
      acc44(38)=Qspval4e1*acc44(26)
      acc44(39)=Qspvae2k2*acc44(17)
      acc44(40)=Qspvae1k2*acc44(28)
      acc44(41)=Qspvak2e1*acc44(29)
      acc44(42)=Qspvae2k1*acc44(2)
      acc44(43)=Qspvak1e2*acc44(25)
      acc44(44)=Qspvae1k1*acc44(20)
      acc44(45)=Qspvak1e1*acc44(21)
      acc44(46)=Qspval5l4*acc44(24)
      acc44(47)=Qspval5k2*acc44(30)
      acc44(48)=Qspval5k1*acc44(16)
      acc44(49)=Qspval4l5*acc44(27)
      acc44(50)=Qspval4k2*acc44(22)
      acc44(51)=Qspval4k1*acc44(8)
      acc44(52)=Qspvak2l4*acc44(23)
      acc44(53)=Qspvak2k1*acc44(14)
      acc44(54)=Qspvak1l5*acc44(19)
      acc44(55)=Qspvak1l4*acc44(5)
      acc44(56)=Qspvak1k2*acc44(6)
      acc44(57)=Qspl4*acc44(12)
      acc44(58)=Qspk2*acc44(3)
      acc44(59)=Qspk1*acc44(4)
      brack=acc44(1)+acc44(31)+acc44(32)+acc44(33)+acc44(34)+acc44(35)+acc44(36&
      &)+acc44(37)+acc44(38)+acc44(39)+acc44(40)+acc44(41)+acc44(42)+acc44(43)+&
      &acc44(44)+acc44(45)+acc44(46)+acc44(47)+acc44(48)+acc44(49)+acc44(50)+ac&
      &c44(51)+acc44(52)+acc44(53)+acc44(54)+acc44(55)+acc44(56)+acc44(57)+acc4&
      &4(58)+acc44(59)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d44h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd44h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d44
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d44 = 0.0_ki
      d44 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d44, ki), aimag(d44), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d44h4l1_qp
