module     p2_gg_httbar_d133h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d133h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd133h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc133(60)
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspl5
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: QspQ
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspl5 = dotproduct(Q,l5)
      Qspk2 = dotproduct(Q,k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      QspQ = dotproduct(Q,Q)
      acc133(1)=abb133(11)
      acc133(2)=abb133(12)
      acc133(3)=abb133(13)
      acc133(4)=abb133(14)
      acc133(5)=abb133(15)
      acc133(6)=abb133(16)
      acc133(7)=abb133(17)
      acc133(8)=abb133(18)
      acc133(9)=abb133(19)
      acc133(10)=abb133(20)
      acc133(11)=abb133(21)
      acc133(12)=abb133(22)
      acc133(13)=abb133(23)
      acc133(14)=abb133(24)
      acc133(15)=abb133(25)
      acc133(16)=abb133(26)
      acc133(17)=abb133(28)
      acc133(18)=abb133(32)
      acc133(19)=abb133(33)
      acc133(20)=abb133(35)
      acc133(21)=abb133(37)
      acc133(22)=abb133(40)
      acc133(23)=abb133(44)
      acc133(24)=abb133(46)
      acc133(25)=abb133(49)
      acc133(26)=abb133(57)
      acc133(27)=abb133(70)
      acc133(28)=abb133(74)
      acc133(29)=abb133(82)
      acc133(30)=abb133(88)
      acc133(31)=abb133(92)
      acc133(32)=abb133(96)
      acc133(33)=abb133(100)
      acc133(34)=abb133(107)
      acc133(35)=abb133(128)
      acc133(36)=acc133(2)*Qspval5e1
      acc133(37)=acc133(4)*Qspval5k2
      acc133(38)=acc133(7)*Qspval5k1
      acc133(39)=acc133(25)*Qspval5l4
      acc133(40)=acc133(26)*Qspl5
      acc133(41)=acc133(29)*Qspk2
      acc133(42)=Qspvae1k2*acc133(20)
      acc133(43)=Qspval4k2*acc133(10)
      acc133(44)=Qspvak1k2*acc133(16)
      acc133(36)=acc133(44)+acc133(43)+acc133(42)+acc133(41)+acc133(40)+acc133(&
      &39)+acc133(19)+acc133(38)+acc133(37)+acc133(36)
      acc133(36)=Qspe2*acc133(36)
      acc133(37)=acc133(1)*Qspval5k2
      acc133(38)=acc133(5)*Qspk2
      acc133(39)=acc133(6)*Qspval5k1
      acc133(40)=acc133(17)*Qspl5
      acc133(41)=acc133(21)*Qspval5e1
      acc133(42)=acc133(24)*Qspval5l4
      acc133(43)=Qspvae2e1*acc133(33)
      acc133(44)=Qspvae1e2*acc133(28)
      acc133(45)=Qspvae2l5*acc133(32)
      acc133(46)=Qspval5e2*acc133(34)
      acc133(47)=Qspvae1l5*acc133(31)
      acc133(48)=Qspvae2l4*acc133(35)
      acc133(49)=Qspval4e2*acc133(27)
      acc133(50)=Qspvae2k2*acc133(3)
      acc133(51)=Qspvak2e2*acc133(13)
      acc133(52)=Qspvak2e1*acc133(15)
      acc133(53)=Qspvae2k1*acc133(22)
      acc133(54)=Qspvak1e2*acc133(23)
      acc133(55)=Qspval4l5*acc133(8)
      acc133(56)=Qspvak2l5*acc133(9)
      acc133(57)=Qspvak2l4*acc133(12)
      acc133(58)=Qspvak2k1*acc133(11)
      acc133(59)=Qspvak1l5*acc133(14)
      acc133(60)=QspQ*acc133(30)
      brack=acc133(18)+acc133(36)+acc133(37)+acc133(38)+acc133(39)+acc133(40)+a&
      &cc133(41)+acc133(42)+acc133(43)+acc133(44)+acc133(45)+acc133(46)+acc133(&
      &47)+acc133(48)+acc133(49)+acc133(50)+acc133(51)+acc133(52)+acc133(53)+ac&
      &c133(54)+acc133(55)+acc133(56)+acc133(57)+acc133(58)+acc133(59)+acc133(6&
      &0)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d133h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd133h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d133
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d133 = 0.0_ki
      d133 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d133, ki), aimag(d133), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d133h4l1_qp
