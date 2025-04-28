module     p2_gg_httbar_d37h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d37h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd37h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc37(41)
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
      acc37(1)=abb37(15)
      acc37(2)=abb37(16)
      acc37(3)=abb37(17)
      acc37(4)=abb37(18)
      acc37(5)=abb37(19)
      acc37(6)=abb37(20)
      acc37(7)=abb37(21)
      acc37(8)=abb37(22)
      acc37(9)=abb37(23)
      acc37(10)=abb37(24)
      acc37(11)=abb37(25)
      acc37(12)=abb37(27)
      acc37(13)=abb37(28)
      acc37(14)=abb37(29)
      acc37(15)=abb37(30)
      acc37(16)=abb37(32)
      acc37(17)=abb37(33)
      acc37(18)=abb37(34)
      acc37(19)=abb37(36)
      acc37(20)=abb37(41)
      acc37(21)=abb37(50)
      acc37(22)=Qspvae2l5*acc37(8)
      acc37(23)=Qspvae1l5*acc37(9)
      acc37(24)=Qspvae2l3*acc37(5)
      acc37(25)=Qspval3e2*acc37(14)
      acc37(26)=Qspvae1l3*acc37(17)
      acc37(27)=Qspval3e1*acc37(18)
      acc37(28)=Qspvak2e2*acc37(13)
      acc37(29)=Qspvak2e1*acc37(3)
      acc37(30)=Qspval5l3*acc37(6)
      acc37(31)=Qspval3l5*acc37(12)
      acc37(32)=Qspval3k2*acc37(19)
      acc37(33)=Qspval3k1*acc37(20)
      acc37(34)=Qspvak2l5*acc37(11)
      acc37(35)=Qspvak2l3*acc37(10)
      acc37(36)=Qspvak2k1*acc37(15)
      acc37(37)=Qspvak1l5*acc37(1)
      acc37(38)=Qspvak1l3*acc37(7)
      acc37(39)=Qspl5*acc37(21)
      acc37(40)=Qspl3*acc37(16)
      acc37(41)=Qspk2*acc37(4)
      brack=acc37(2)+acc37(22)+acc37(23)+acc37(24)+acc37(25)+acc37(26)+acc37(27&
      &)+acc37(28)+acc37(29)+acc37(30)+acc37(31)+acc37(32)+acc37(33)+acc37(34)+&
      &acc37(35)+acc37(36)+acc37(37)+acc37(38)+acc37(39)+acc37(40)+acc37(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d37h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd37h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d37
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d37 = 0.0_ki
      d37 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d37, ki), aimag(d37), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d37h8l1_qp
