module     p2_gg_httbar_d129h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d129h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd129h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc129(43)
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl5 = dotproduct(Q,l5)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc129(1)=abb129(13)
      acc129(2)=abb129(14)
      acc129(3)=abb129(15)
      acc129(4)=abb129(16)
      acc129(5)=abb129(17)
      acc129(6)=abb129(18)
      acc129(7)=abb129(19)
      acc129(8)=abb129(20)
      acc129(9)=abb129(21)
      acc129(10)=abb129(22)
      acc129(11)=abb129(23)
      acc129(12)=abb129(26)
      acc129(13)=abb129(27)
      acc129(14)=abb129(29)
      acc129(15)=abb129(34)
      acc129(16)=abb129(36)
      acc129(17)=abb129(38)
      acc129(18)=abb129(39)
      acc129(19)=abb129(41)
      acc129(20)=abb129(86)
      acc129(21)=abb129(161)
      acc129(22)=abb129(179)
      acc129(23)=Qspvae2l5*acc129(14)
      acc129(24)=Qspvae1l5*acc129(8)
      acc129(25)=Qspvae2l3*acc129(15)
      acc129(26)=Qspval3e2*acc129(17)
      acc129(27)=Qspvae1l3*acc129(19)
      acc129(28)=Qspval3e1*acc129(16)
      acc129(29)=Qspvak2e2*acc129(13)
      acc129(30)=Qspvak2e1*acc129(2)
      acc129(31)=Qspval5l3*acc129(5)
      acc129(32)=Qspval3l5*acc129(4)
      acc129(33)=Qspval3k2*acc129(3)
      acc129(34)=Qspval3k1*acc129(21)
      acc129(35)=Qspvak2l5*acc129(1)
      acc129(36)=Qspvak2l3*acc129(6)
      acc129(37)=Qspvak2k1*acc129(10)
      acc129(38)=Qspvak1l5*acc129(11)
      acc129(39)=Qspvak1l3*acc129(22)
      acc129(40)=Qspl5*acc129(18)
      acc129(41)=Qspl3*acc129(20)
      acc129(42)=Qspk2*acc129(9)
      acc129(43)=QspQ*acc129(12)
      brack=acc129(7)+acc129(23)+acc129(24)+acc129(25)+acc129(26)+acc129(27)+ac&
      &c129(28)+acc129(29)+acc129(30)+acc129(31)+acc129(32)+acc129(33)+acc129(3&
      &4)+acc129(35)+acc129(36)+acc129(37)+acc129(38)+acc129(39)+acc129(40)+acc&
      &129(41)+acc129(42)+acc129(43)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d129h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd129h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d129
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d129 = 0.0_ki
      d129 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d129, ki), aimag(d129), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d129h8l1_qp
