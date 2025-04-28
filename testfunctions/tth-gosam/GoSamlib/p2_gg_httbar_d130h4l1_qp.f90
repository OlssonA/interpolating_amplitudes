module     p2_gg_httbar_d130h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d130h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd130h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc130(43)
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl4
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl4 = dotproduct(Q,l4)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc130(1)=abb130(13)
      acc130(2)=abb130(14)
      acc130(3)=abb130(15)
      acc130(4)=abb130(16)
      acc130(5)=abb130(17)
      acc130(6)=abb130(18)
      acc130(7)=abb130(19)
      acc130(8)=abb130(20)
      acc130(9)=abb130(21)
      acc130(10)=abb130(22)
      acc130(11)=abb130(23)
      acc130(12)=abb130(26)
      acc130(13)=abb130(27)
      acc130(14)=abb130(29)
      acc130(15)=abb130(34)
      acc130(16)=abb130(36)
      acc130(17)=abb130(38)
      acc130(18)=abb130(39)
      acc130(19)=abb130(41)
      acc130(20)=abb130(86)
      acc130(21)=abb130(161)
      acc130(22)=abb130(179)
      acc130(23)=Qspvae2l4*acc130(14)
      acc130(24)=Qspvae1l4*acc130(8)
      acc130(25)=Qspvae2l3*acc130(15)
      acc130(26)=Qspval3e2*acc130(17)
      acc130(27)=Qspvae1l3*acc130(19)
      acc130(28)=Qspval3e1*acc130(16)
      acc130(29)=Qspvak2e2*acc130(13)
      acc130(30)=Qspvak2e1*acc130(2)
      acc130(31)=Qspval4l3*acc130(5)
      acc130(32)=Qspval3l4*acc130(4)
      acc130(33)=Qspval3k2*acc130(3)
      acc130(34)=-Qspval3k1*acc130(21)
      acc130(35)=Qspvak2l4*acc130(1)
      acc130(36)=Qspvak2l3*acc130(6)
      acc130(37)=Qspvak2k1*acc130(10)
      acc130(38)=Qspvak1l4*acc130(11)
      acc130(39)=-Qspvak1l3*acc130(22)
      acc130(40)=Qspl4*acc130(18)
      acc130(41)=Qspl3*acc130(20)
      acc130(42)=Qspk2*acc130(9)
      acc130(43)=QspQ*acc130(12)
      brack=acc130(7)+acc130(23)+acc130(24)+acc130(25)+acc130(26)+acc130(27)+ac&
      &c130(28)+acc130(29)+acc130(30)+acc130(31)+acc130(32)+acc130(33)+acc130(3&
      &4)+acc130(35)+acc130(36)+acc130(37)+acc130(38)+acc130(39)+acc130(40)+acc&
      &130(41)+acc130(42)+acc130(43)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d130h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd130h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d130
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d130 = 0.0_ki
      d130 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d130, ki), aimag(d130), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d130h4l1_qp
